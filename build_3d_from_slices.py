"""Baue aus echten 2D-CT-Slices ein 3D-Volumen / 3D-Modell einer Batteriezelle.

Dieses Skript ist REIN GEOMETRISCH -- es ist keine KI-Anwendung. Es nimmt die
bereits vorhandenen, realen Schnittbilder (Slices) einer Zelle, stapelt sie in
der richtigen Reihenfolge entlang der Zellachse (z), skaliert korrekt in der
Tiefe (physikalischer Slice-Abstand vs. In-Plane-Pixelgroesse) und exportiert
ein ansehbares 3D-Modell.

Ablauf
------
1. Ordner mit Slices auswaehlen (GUI-Dialog) oder per --input uebergeben.
2. Alle Bilder laden, nach Tiefe sortieren (aus dem Dateinamen, in mm).
3. Zu einem Volumen [D, H, W] stapeln, in-plane optional heruntergerechnet.
4. Physikalische Voxelgroessen bestimmen:
     - in-plane: aus dem Zellformat (18650 -> 14.4 um, 2170 -> 16.4 um, 4680 -> 35 um)
     - in der Tiefe: aus dem medianen Abstand der abs_depth-Werte (mm)
   Daraus ergibt sich das Streckungsverhaeltnis in z.
   Die Slices werden an ihre ECHTEN Tiefen gesetzt (aus abs_depth), damit der
   Aspekt auch bei ungleichmaessiger Abtastung stimmt (--no-true-depth schaltet
   das ab).
5. Exportieren:
     - volume.tif        3D-Stack der Rohslices (Fiji/ImageJ, napari, 3D Slicer)
     - volume.npy        rohes Array fuer eigene Weiterverarbeitung
     - view_3d.html       interaktiver Viewer: Volume-Rendering (3D-Schnitt) + 2D-Schnitte
     - surface.stl        Isoflaeche als Mesh (Blender, Windows 3D-Viewer, MeshLab)
     - preview_ortho.png  drei orthogonale Schnittebenen (physikalisch korrekter Aspekt)
     - metadata.json      alle verwendeten Parameter
   Optional:
     - volume_geo.tif     aspekt-korrektes (resampletes) Volumen (--isotropic)
     - preview_3d.png     gerenderte Isoflaeche (--render-3d)

Im Browser ansehen
------------------
  view_3d.html per Doppelklick oeffnen -- kein Server, kein CDN, keine
  Internetverbindung noetig (Three.js ist in der Datei eingebettet). Links die
  Zelle als massiver 3D-Koerper (Volume-Rendering); der Schieber schiebt eine
  Schnittebene hinein und schneidet den Koerper auf, sodass die Schnittflaeche
  die Innenstruktur zeigt. Rechts der exakte 2D-Grauwert-Schnitt. Ebene
  (axial/koronal/sagittal) waehlen; beide Ansichten sind synchronisiert.

  Die Three.js-Bibliothek liegt in reconstruction_3d/lib/ (three.min.js,
  OrbitControls.js) und wird beim Erzeugen eingebettet. Das 3D benoetigt WebGL2;
  fehlt es (oder die Lib), bleibt die 3D-Ansicht leer, die 2D-Schnitte laufen weiter.

Beispiele
---------
  # GUI-Ordnerauswahl, alle Standardausgaben:
  python reconstruction_3d/build_3d_from_slices.py

  # Ordner direkt, in-plane auf max. 384 px, nur jeden 2. Slice, Mesh + isotrop:
  python reconstruction_3d/build_3d_from_slices.py \
      --input "D:/scans/18650/slices/cellA/radial_images" \
      --max-inplane 384 --max-slices 200 --isotropic
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# --- Eigenstaendige Helfer (frueher aus battery_metadata; hier inline, damit
#     dieses Skript ohne das uebrige Projekt lauffaehig ist) -----------------

def _voxel_size_um_for_format(cell_format: str) -> Optional[float]:
    """In-plane-Voxelgroesse in um je Zellformat (None, falls unbekannt)."""
    f = str(cell_format)
    if "18650" in f:
        return 14.4
    if "2170" in f:
        return 16.4
    if "4680" in f:
        return 35.0
    return None


def _abs_depth_from_name(filename: str) -> float:
    """Absolute Slice-Tiefe (mm) aus dem Dateinamen: letzte Zahl vor der Endung."""
    import re
    name = os.path.splitext(os.path.basename(filename))[0]
    try:
        return float(name.split("_")[-1])
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.?\d+", name)
        return float(nums[-1]) if nums else 0.0


IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------------
# Slices finden, sortieren, laden
# ---------------------------------------------------------------------------

def _list_image_files(folder: str, recursive: bool) -> List[str]:
    files: List[str] = []
    if recursive:
        for root, _dirs, names in os.walk(folder):
            for n in names:
                if n.lower().endswith(IMAGE_EXTS):
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(folder):
            p = os.path.join(folder, n)
            if os.path.isfile(p) and n.lower().endswith(IMAGE_EXTS):
                files.append(p)
    return files


def _depth_of(path: str) -> float:
    """Tiefe (in mm) aus dem Dateinamen."""
    return _abs_depth_from_name(path)


def _sorted_slices(files: List[str]) -> Tuple[List[str], np.ndarray]:
    pairs = sorted(((p, _depth_of(p)) for p in files), key=lambda t: t[1])
    paths = [p for p, _ in pairs]
    depths = np.array([d for _, d in pairs], dtype=np.float64)
    return paths, depths


def _subsample(paths: List[str], depths: np.ndarray, max_slices: int):
    if max_slices and 0 < max_slices < len(paths):
        idx = np.linspace(0, len(paths) - 1, max_slices).round().astype(int)
        idx = np.unique(idx)
        paths = [paths[i] for i in idx]
        depths = depths[idx]
    return paths, depths


def _read_gray(path: str) -> np.ndarray:
    """Liest ein Bild als uint8-Graustufen. Nutzt cv2, faellt sonst auf imageio zurueck."""
    if cv2 is not None:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(f"Kann Bild nicht oeffnen: {path}")
        return g
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Zum Laden der Bilder wird 'opencv-python' ODER 'imageio' benoetigt. "
            "Bitte eines installieren (pip install opencv-python) oder das Skript "
            "mit dem Projekt-venv ausfuehren."
        ) from e
    arr = np.asarray(imageio.imread(path))
    if arr.ndim == 3:                       # Farbbild -> Graustufen
        arr = arr[..., :3].mean(axis=2)
    if arr.dtype == np.uint16:              # 16-bit -> 8-bit
        arr = arr / 257.0
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_gray(img: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
    """Skaliert ein Graustufenbild. Nutzt cv2, sonst skimage, sonst PIL."""
    if img.shape[:2] == (tgt_h, tgt_w):
        return img
    if cv2 is not None:
        return cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
    try:
        from skimage.transform import resize as sk_resize  # type: ignore
        out = sk_resize(img, (tgt_h, tgt_w), order=1, anti_aliasing=True,
                        preserve_range=True)
        return out.astype(np.uint8)
    except Exception:
        from PIL import Image  # type: ignore
        return np.asarray(Image.fromarray(img).resize((tgt_w, tgt_h), Image.BILINEAR))


def _load_volume(paths: List[str], max_inplane: int) -> Tuple[np.ndarray, float]:
    """Laedt die Slices in ein Volumen [D, H, W] (uint8, Graustufen).

    Gibt (volume, downsample_factor) zurueck. downsample_factor > 1, wenn
    in-plane heruntergerechnet wurde (fuer die korrekte Voxelgroesse).
    Funktioniert mit OpenCV; ohne OpenCV wird imageio/skimage/PIL genutzt.
    """
    # Zielgroesse aus dem ersten Bild bestimmen.
    first = _read_gray(paths[0])
    h0, w0 = first.shape[:2]

    factor = 1.0
    tgt_h, tgt_w = h0, w0
    if max_inplane and max(h0, w0) > max_inplane:
        factor = max(h0, w0) / float(max_inplane)
        tgt_h = int(round(h0 / factor))
        tgt_w = int(round(w0 / factor))

    vol = np.empty((len(paths), tgt_h, tgt_w), dtype=np.uint8)
    for i, p in enumerate(paths):
        g = _resize_gray(_read_gray(p), tgt_h, tgt_w)
        vol[i] = g
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            print(f"  geladen: {i + 1}/{len(paths)} Slices", end="\r")
    print()
    return vol, factor


# ---------------------------------------------------------------------------
# Physikalische Voxelgroessen
# ---------------------------------------------------------------------------

def _infer_voxel_um(path_hint: str, override: Optional[float]) -> Tuple[float, str]:
    if override is not None:
        return float(override), "override"
    low = path_hint.replace("\\", "/").lower()
    for fmt in ("18650", "2170", "4680"):
        if fmt in low:
            vs = _voxel_size_um_for_format(fmt)
            if vs:
                return float(vs), f"Format {fmt}"
    return 1.0, "Fallback 1.0 (unbekannt)"


def _z_step_mm(depths: np.ndarray) -> Optional[float]:
    if depths.size < 2:
        return None
    diffs = np.diff(depths)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _save_tiff(path: str, vol: np.ndarray) -> None:
    try:
        import tifffile  # type: ignore
        tifffile.imwrite(path, vol)
    except Exception:
        import imageio.v2 as imageio  # type: ignore
        imageio.mimwrite(path, list(vol))


def write_binary_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Schreibt ein binaeres STL (ohne externe Abhaengigkeit).

    verts: [V, 3] float, faces: [F, 3] int (Indizes in verts).
    """
    tris = verts[faces]                      # [F, 3, 3]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = (n / ln).astype("<f4")

    rec = np.dtype([("n", "<f4", 3), ("v0", "<f4", 3),
                    ("v1", "<f4", 3), ("v2", "<f4", 3), ("attr", "<u2")])
    data = np.zeros(faces.shape[0], dtype=rec)
    data["n"] = n
    data["v0"] = v0.astype("<f4")
    data["v1"] = v1.astype("<f4")
    data["v2"] = v2.astype("<f4")

    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", faces.shape[0]))
        f.write(data.tobytes())


def build_mesh(vol: np.ndarray, spacing_zyx: Tuple[float, float, float],
               iso_level: Optional[float]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    try:
        from skimage import measure  # type: ignore
        from skimage.filters import threshold_otsu  # type: ignore
    except Exception:
        print("  [Mesh] scikit-image nicht verfuegbar -> ueberspringe STL.")
        return None

    v = vol.astype(np.float32)
    level = float(iso_level) if iso_level is not None else float(threshold_otsu(v))
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(
            v, level=level, spacing=spacing_zyx,
        )
    except Exception as e:  # z.B. leere Isoflaeche
        print(f"  [Mesh] marching_cubes fehlgeschlagen: {e}")
        return None
    return verts, faces, level


def save_orthoviews(path: str, vol: np.ndarray,
                    spacing_zyx: Tuple[float, float, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sz, sy, sx = spacing_zyx
    d, h, w = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial (Blick entlang z) -> Ebene (y, x)
    axes[0].imshow(vol[d // 2], cmap="gray", aspect=sy / sx)
    axes[0].set_title(f"axial  z={d // 2}")

    # Koronal (Blick entlang y) -> Ebene (z, x); z muss gestreckt werden
    axes[1].imshow(vol[:, h // 2, :], cmap="gray", aspect=sz / sx)
    axes[1].set_title(f"koronal  y={h // 2}")

    # Sagittal (Blick entlang x) -> Ebene (z, y)
    axes[2].imshow(vol[:, :, w // 2], cmap="gray", aspect=sz / sy)
    axes[2].set_title(f"sagittal  x={w // 2}")

    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_3d(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Bei sehr grossen Meshes eine Teilmenge der Dreiecke zeichnen (Speed).
    max_faces = 120_000
    f = faces
    if f.shape[0] > max_faces:
        sel = np.linspace(0, f.shape[0] - 1, max_faces).astype(int)
        f = f[sel]

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[f], alpha=1.0)
    mesh.set_edgecolor("none")
    mesh.set_facecolor((0.6, 0.6, 0.65))
    ax.add_collection3d(mesh)

    ax.set_xlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_box_aspect((np.ptp(verts[:, 2]), np.ptp(verts[:, 1]), np.ptp(verts[:, 0])))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_isotropic(vol: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    """Resampled das Volumen so, dass die z-Voxel den in-plane-Voxeln entsprechen."""
    from scipy.ndimage import zoom  # type: ignore
    sz, sy, sx = spacing_zyx
    target = min(sy, sx)
    zoom_factors = (sz / target, sy / target, sx / target)
    return zoom(vol, zoom_factors, order=1).astype(vol.dtype)


def _resample_true_depth(vol: np.ndarray, depths: np.ndarray, sxy_mm: float,
                         max_z: int = 384) -> Tuple[np.ndarray, float, float]:
    """Setzt die Slices an ihre ECHTEN Tiefen und resampled auf ein gleichmaessiges z-Gitter.

    Behebt den Aspekt-Fehler bei ungleichmaessig abgetasteten Slices: statt alle
    Slices gleichabstaendig zu stapeln, werden sie entsprechend ihrer realen
    abs_depth-Werte platziert und linear auf ein regelmaessiges Gitter ueber den
    tatsaechlichen Tiefenbereich interpoliert (Ziel ~ isotrop, gedeckelt auf max_z).

    Returns (vol_uniform [Dz, H, W] uint8, sz_mm, z0_mm).
    """
    from scipy.interpolate import interp1d  # type: ignore

    d = np.asarray(depths, dtype=np.float64)
    order = np.argsort(d)
    d = d[order]
    v = vol[order]
    # Doppelte Tiefen entfernen (interp1d braucht streng monotone x-Werte).
    uniq, idx = np.unique(d, return_index=True)
    if uniq.size != d.size:
        d = uniq
        v = v[idx]
    dmin, dmax = float(d[0]), float(d[-1])
    rng = max(dmax - dmin, 1e-6)
    n_iso = int(round(rng / max(sxy_mm, 1e-6))) + 1
    n_z = int(max(len(d), min(int(max_z), n_iso)))
    z_dst = np.linspace(dmin, dmax, n_z)
    f = interp1d(d, v.astype(np.float32), axis=0, kind="linear", assume_sorted=True)
    vol_z = np.clip(f(z_dst), 0, 255).astype(np.uint8)
    sz_mm = rng / (n_z - 1) if n_z > 1 else sxy_mm
    return vol_z, float(sz_mm), dmin


def _downsample_vol(vol: np.ndarray, spacing_zyx: Tuple[float, float, float],
                    max_inplane: int, max_z: int):
    """Verkleinert ein Volumen auf Ziel-Kantenlaengen und skaliert die Voxelabstaende mit."""
    from scipy.ndimage import zoom  # type: ignore
    D, H, W = vol.shape
    fz = min(1.0, float(max_z) / D) if max_z else 1.0
    fin = min(1.0, float(max_inplane) / max(H, W)) if max_inplane else 1.0
    if fz >= 1.0 and fin >= 1.0:
        return vol, spacing_zyx
    out = np.clip(zoom(vol, (fz, fin, fin), order=1), 0, 255).astype(np.uint8)
    sz, sy, sx = spacing_zyx
    Dn, Hn, Wn = out.shape
    sp = (sz * D / max(1, Dn), sy * H / max(1, Hn), sx * W / max(1, Wn))
    return out, sp


def build_shell_mesh(vol: np.ndarray, spacing_zyx: Tuple[float, float, float],
                     max_inplane: int = 96, max_z: int = 200):
    """Baut ein schlankes Aussenhuellen-Mesh (die Can/Aussenkontur) fuer die 3D-Ansicht.

    Es wird NICHT jede Wickellage als Flaeche erzeugt (das waere riesig), sondern
    nur die Aussenhuelle: Schwellwert -> pro Slice Loecher fuellen -> groesste
    Komponente -> marching_cubes. Dadurch bleibt das eingebettete Mesh klein.

    Returns (verts [V,3] in mm (z,y,x), faces [F,3]) oder None.
    """
    try:
        from skimage import measure  # type: ignore
        from skimage.filters import threshold_otsu  # type: ignore
        from scipy import ndimage  # type: ignore
    except Exception:
        return None
    v, sp = _downsample_vol(vol, spacing_zyx, max_inplane, max_z)
    try:
        thr = max(8.0, 0.4 * float(threshold_otsu(v)))
    except Exception:
        thr = 12.0
    mask = v > thr
    if not mask.any():
        return None
    for k in range(mask.shape[0]):
        mask[k] = ndimage.binary_fill_holes(mask[k])
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab, dtype=np.int64), lab, index=range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    try:
        verts, faces, _n, _v = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=sp)
    except Exception:
        return None
    return verts.astype(np.float32), faces.astype(np.int64)


_HTML_VIEWER = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CT-Viewer</title>
<style>
  html,body{margin:0;min-height:100%;background:#0f1115;color:#d7dae0;
            font-family:system-ui,sans-serif}
  #bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 14px;font-size:14px}
  #views{display:flex;flex-wrap:wrap;gap:10px;padding:0 10px 16px}
  .pane{flex:1 1 360px;min-width:290px;height:540px;background:#000;border-radius:8px;
        overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative}
  .tag{position:absolute;top:8px;left:10px;font-size:12px;opacity:.65;z-index:2;pointer-events:none}
  #cv3{width:100%;height:100%;display:block}
  #cv{max-width:100%;max-height:100%}
  .pill{background:#1b1f27;border-radius:6px;padding:3px 9px}
  select,button{background:#1b1f27;color:#d7dae0;border:1px solid #333;
                border-radius:6px;padding:4px 9px;cursor:pointer;font-size:14px}
  input[type=range]{width:300px;vertical-align:middle}
</style>
</head>
<body>
<div id="bar">
  <span class="pill" id="title">__TITLE__</span>
  <label>Ebene:
    <select id="axis">
      <option value="0">axial (quer, z)</option>
      <option value="1">koronal (y)</option>
      <option value="2">sagittal (x)</option>
    </select>
  </label>
  <input id="slider" type="range" min="0" max="1" value="0">
  <span class="pill" id="pos"></span>
  <button id="play">&#9654; Durchlauf</button>
</div>
<div id="views">
  <div class="pane" id="pane3d"><span class="tag">3D &middot; ziehen: drehen &middot; scrollen: zoomen</span><canvas id="cv3"></canvas></div>
  <div class="pane" id="pane2d"><span class="tag">2D-Schnitt</span><canvas id="cv"></canvas></div>
</div>
<script>
(function(){
  const DIMS=__DIMS__, SP=__SPACING__, Z0=__Z0__;
  function b64(s){const b=atob(s);const u=new Uint8Array(b.length);
    for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
  const VOL=b64("__VOL_B64__");
  const D=DIMS[0],H=DIMS[1],W=DIMS[2], SZ=SP[0],SY=SP[1],SX=SP[2];
  const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
  const off=document.createElement('canvas'), octx=off.getContext('2d');
  const axisSel=document.getElementById('axis'), slider=document.getElementById('slider'),
        posEl=document.getElementById('pos'), playBtn=document.getElementById('play');
  const MAXBOX=520;
  function info(axis){
    if(axis===0) return {rows:H,cols:W,rowS:SY,colS:SX,n:D,get:(k,r,c)=>VOL[k*H*W+r*W+c]};
    if(axis===1) return {rows:D,cols:W,rowS:SZ,colS:SX,n:H,get:(k,r,c)=>VOL[r*H*W+k*W+c]};
    return {rows:D,cols:H,rowS:SZ,colS:SY,n:W,get:(k,r,c)=>VOL[r*H*W+c*W+k]};
  }
  function draw(){
    const axis=+axisSel.value, k=+slider.value, s=info(axis);
    off.width=s.cols; off.height=s.rows;
    const img=octx.createImageData(s.cols,s.rows), dd=img.data;
    for(let r=0;r<s.rows;r++) for(let c=0;c<s.cols;c++){
      const v=s.get(k,r,c), i=(r*s.cols+c)*4; dd[i]=dd[i+1]=dd[i+2]=v; dd[i+3]=255;
    }
    octx.putImageData(img,0,0);
    const pw=s.cols*s.colS, ph=s.rows*s.rowS, sc=Math.min(MAXBOX/pw, MAXBOX/ph);
    cv.width=Math.max(1,Math.round(pw*sc)); cv.height=Math.max(1,Math.round(ph*sc));
    ctx.imageSmoothingEnabled=true;
    if(axis===0){ ctx.setTransform(1,0,0,1,0,0); }
    else { ctx.setTransform(1,0,0,-1,0,cv.height); }
    ctx.drawImage(off,0,0,s.cols,s.rows,0,0,cv.width,cv.height);
    ctx.setTransform(1,0,0,1,0,0);
    let label='Schnitt '+(k+1)+'/'+s.n;
    if(axis===0) label+='  ·  z = '+(Z0+k*SZ).toFixed(2)+' mm';
    posEl.textContent=label;
    if(window.on3D) window.on3D(axis,k);
  }
  function setAxis(){ const s=info(+axisSel.value); slider.max=s.n-1;
    slider.value=Math.floor((s.n-1)/2); draw(); }
  axisSel.onchange=setAxis; slider.oninput=draw;
  let timer=null;
  playBtn.onclick=()=>{
    if(timer){clearInterval(timer);timer=null;playBtn.innerHTML='&#9654; Durchlauf';return;}
    playBtn.innerHTML='&#10073;&#10073; Stop';
    timer=setInterval(()=>{let v=+slider.value+1; if(v>+slider.max)v=0; slider.value=v; draw();},80);
  };
  window.CT={DIMS,SP,VOL,cv,refresh:draw};
  setAxis();
})();
</script>
<script>__THREE_JS__</script>
<script>__ORBIT_JS__</script>
<script>
(function(){
  const C=window.CT, pane=document.getElementById('pane3d');
  if(!C){ pane.style.display='none'; return; }
  function fail(msg){ pane.innerHTML='<span style="opacity:.7;font-size:13px;padding:14px;'+
    'text-align:center">'+msg+'<br>Die 2D-Schnitte rechts funktionieren weiterhin.</span>'; }
  if(!window.THREE || !THREE.OrbitControls){ fail('3D-Bibliothek nicht geladen.'); return; }

  const cv3=document.getElementById('cv3');
  const D=C.DIMS[0],H=C.DIMS[1],W=C.DIMS[2], SZ=C.SP[0],SY=C.SP[1],SX=C.SP[2];
  // Three-Achsen: X = x (W), Y = z/Hoehe (D), Z = y (H)
  const EX=W*SX, EY=D*SZ, EZ=H*SY;

  const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0b0d11);
  const cam=new THREE.PerspectiveCamera(45, pane.clientWidth/Math.max(1,pane.clientHeight), 0.01, 1e7);
  const rnd=new THREE.WebGLRenderer({canvas:cv3, antialias:true});
  rnd.setPixelRatio(devicePixelRatio);
  function resize(){ rnd.setSize(pane.clientWidth, pane.clientHeight, false);
    cam.aspect=pane.clientWidth/Math.max(1,pane.clientHeight); cam.updateProjectionMatrix(); }
  resize(); addEventListener('resize', resize);
  const ctrl=new THREE.OrbitControls(cam, rnd.domElement); ctrl.enableDamping=true;

  if(!rnd.capabilities.isWebGL2 || !THREE.DataTexture3D){
    fail('Volumen-Rendering benoetigt WebGL2 (in diesem Browser nicht verfuegbar).'); return;
  }

  // Volumen als 3D-Textur (R8). Datenlayout C.VOL = z*H*W + y*W + x -> (W,H,D).
  const tex=new THREE.DataTexture3D(C.VOL, W, H, D);
  tex.format=THREE.RedFormat; tex.type=THREE.UnsignedByteType;
  tex.minFilter=THREE.LinearFilter; tex.magFilter=THREE.LinearFilter;
  tex.wrapR=tex.wrapS=tex.wrapT=THREE.ClampToEdgeWrapping;
  tex.unpackAlignment=1; tex.needsUpdate=true;

  const uniforms={
    uVol:{value:tex},
    uHalf:{value:new THREE.Vector3(EX/2,EY/2,EZ/2)},
    uAxis:{value:1}, uClip:{value:EY/2},
    uSteps:{value:Math.min(640, Math.max(160, D+H+W))},
    uLo:{value:0.10}, uHi:{value:0.92}, uDensity:{value:0.16}
  };
  const vert='out vec3 vLocal;\\nvoid main(){ vLocal=position;'+
    ' gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }';
  const frag=[
    'precision highp float;','precision highp sampler3D;',
    'in vec3 vLocal; out vec4 fragColor;',
    'uniform highp sampler3D uVol; uniform vec3 uHalf; uniform int uAxis;',
    'uniform float uClip; uniform int uSteps; uniform float uLo,uHi,uDensity;',
    'vec3 toUVW(vec3 p){ return vec3(p.x/(2.0*uHalf.x)+0.5, p.z/(2.0*uHalf.z)+0.5, p.y/(2.0*uHalf.y)+0.5); }',
    'void main(){',
    ' vec3 ro=cameraPosition; vec3 rd=normalize(vLocal-ro);',
    ' vec3 t1=(-uHalf-ro)/rd, t2=(uHalf-ro)/rd;',
    ' vec3 tmn=min(t1,t2), tmx=max(t1,t2);',
    ' float tN=max(max(max(tmn.x,tmn.y),tmn.z),0.0);',
    ' float tF=min(min(tmx.x,tmx.y),tmx.z);',
    ' if(tN>=tF) discard;',
    ' float rda = uAxis==0?rd.x:(uAxis==1?rd.y:rd.z);',
    ' float roa = uAxis==0?ro.x:(uAxis==1?ro.y:ro.z);',
    ' if(abs(rda)>1e-6){ float tp=(uClip-roa)/rda; if(rda>0.0) tF=min(tF,tp); else tN=max(tN,tp); }',
    ' else { if(roa>uClip) discard; }',
    ' if(tN>=tF) discard;',
    ' float dt=(tF-tN)/float(uSteps);',
    ' vec3 col=vec3(0.0); float a=0.0;',
    ' for(int i=0;i<1024;i++){ if(i>=uSteps) break;',
    '   float tt=tN+(float(i)+0.5)*dt; vec3 p=ro+rd*tt;',
    '   float v=texture(uVol, toUVW(p)).r;',
    '   float op=clamp((v-uLo)/max(1e-3,(uHi-uLo)),0.0,1.0)*uDensity;',
    '   col += (1.0-a)*op*vec3(v); a += (1.0-a)*op;',
    '   if(a>0.985) break; }',
    ' if(a<0.01) discard;',
    ' fragColor=vec4(col,a);',
    '}'
  ].join('\\n');

  const mat=new THREE.ShaderMaterial({uniforms:uniforms, vertexShader:vert, fragmentShader:frag,
    glslVersion:THREE.GLSL3, side:THREE.BackSide, transparent:true, depthWrite:false,
    blending:THREE.CustomBlending, blendEquation:THREE.AddEquation,
    blendSrc:THREE.OneFactor, blendDst:THREE.OneMinusSrcAlphaFactor});
  const box=new THREE.BoxGeometry(EX,EY,EZ);
  scene.add(new THREE.Mesh(box, mat));
  // Bezugsrahmen (Quader-Kanten), auch sichtbar falls der Shader leer bleibt
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(box),
    new THREE.LineBasicMaterial({color:0x33405a})));

  // Slider -> Schnittebene. Es wird die Haelfte oberhalb der Ebene weggeschnitten.
  window.on3D=function(axis,k){
    if(axis===0){ uniforms.uAxis.value=1; uniforms.uClip.value=k*SZ-EY/2; }      // axial  -> Hoehe (Y)
    else if(axis===1){ uniforms.uAxis.value=2; uniforms.uClip.value=k*SY-EZ/2; } // koronal -> Z
    else { uniforms.uAxis.value=0; uniforms.uClip.value=k*SX-EX/2; }             // sagittal-> X
  };

  const R=Math.max(EX,EY,EZ);
  cam.position.set(R*0.9, R*0.6, R*1.7);
  ctrl.target.set(0,0,0); ctrl.update();
  (function loop(){ requestAnimationFrame(loop); ctrl.update(); rnd.render(scene,cam); })();
  C.refresh();
})();
</script>
</body>
</html>
"""


def write_html_viewer(path: str, vol: np.ndarray,
                      spacing_zyx: Tuple[float, float, float],
                      z0_mm: float, title: str = "CT") -> None:
    """Schreibt einen eigenstaendigen Viewer: Volume-Rendering (3D) + 2D-Schnitte.

    Eingebettet werden das Grauwert-Volumen und die Three.js-Bibliothek selbst --
    alles in der HTML. Links wird die Zelle per Volume-Rendering (3D-Textur +
    Raymarching) als massiver Koerper dargestellt; der Slider schiebt eine
    Schnittebene hinein und schneidet den Koerper auf. Rechts der exakte 2D-
    Grauwert-Schnitt. Laeuft per Doppelklick offline (kein Server, kein CDN).
    Das 3D benoetigt WebGL2; faellt es aus, bleiben die 2D-Schnitte nutzbar.
    """
    import base64
    vol = np.ascontiguousarray(vol, dtype=np.uint8)
    D, H, W = vol.shape
    vol_b64 = base64.b64encode(vol.tobytes()).decode("ascii")
    sz, sy, sx = spacing_zyx

    # Three.js + OrbitControls direkt einbetten (offline). Fehlen die Dateien,
    # bleibt das 3D leer (die 2D-Schnitte funktionieren weiterhin).
    libdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

    def _read_lib(name: str) -> str:
        try:
            with open(os.path.join(libdir, name), "r", encoding="utf-8") as fh:
                # Verhindert ein verfruehtes Schliessen des <script>-Tags.
                return fh.read().replace("</script", "<\\/script")
        except OSError:
            return ""

    three_js = _read_lib("three.min.js")
    orbit_js = _read_lib("OrbitControls.js")

    html = (_HTML_VIEWER
            .replace("__THREE_JS__", three_js)
            .replace("__ORBIT_JS__", orbit_js)
            .replace("__DIMS__", f"[{D},{H},{W}]")
            .replace("__SPACING__", f"[{sz:.6f},{sy:.6f},{sx:.6f}]")
            .replace("__Z0__", f"{float(z0_mm):.6f}")
            .replace("__TITLE__", title)
            .replace("__VOL_B64__", vol_b64))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Ordnerauswahl
# ---------------------------------------------------------------------------

def _pick_folder_gui() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Ordner mit CT-Slices auswaehlen")
    root.destroy()
    return folder or None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="3D-Volumen/Modell aus 2D-CT-Slices bauen.")
    p.add_argument("--input", default="C:\\Users\\larsr\\Documents\\Uni\\Masterarbeit\\Beispielbilder\\18650_LithiumIon_EVE\\1769\\radial_images", help="Ordner mit Slices (sonst GUI-Dialog).")
    p.add_argument("--output", default="C:\\Users\\larsr\\Documents\\Uni\\Masterarbeit\\Masterarbeit_3D\\reconstruction_3d", help="Ausgabeordner (Standard: <input>/../<name>_3d).")
    p.add_argument("--recursive", action="store_true", help="Auch Unterordner nach Bildern durchsuchen.")
    p.add_argument("--max-inplane", type=int, default=256,
                   help="In-Plane auf max. diese Kantenlaenge herunterrechnen (0 = aus).")
    p.add_argument("--max-slices", type=int, default=0,
                   help="Auf hoechstens so viele Slices ausduennen (0 = alle).")
    p.add_argument("--voxel-um", type=float, default=None,
                   help="In-Plane-Voxelgroesse in um (Standard: aus Format ableiten).")
    p.add_argument("--z-um", type=float, default=None,
                   help="Slice-Abstand in um (Standard: aus abs_depth ableiten).")
    p.add_argument("--iso-level", type=float, default=None,
                   help="Schwellwert fuer die Mesh-Isoflaeche (Standard: Otsu).")
    p.add_argument("--isotropic", action="store_true",default=True, help="Zusaetzlich isotropes Volumen speichern.")
    p.add_argument("--render-3d", action="store_true",default=True, help="Zusaetzlich ein 3D-Vorschau-PNG rendern.")
    p.add_argument("--no-true-depth", action="store_true",
                   help="Slices gleichabstaendig stapeln statt an echten Tiefen "
                        "(nicht empfohlen bei ungleichmaessiger Abtastung).")
    p.add_argument("--html-max-inplane", type=int, default=160,
                   help="In-Plane-Kantenlaenge des im HTML eingebetteten Volumens "
                        "(hoeher = schaerfer, groessere Datei).")
    p.add_argument("--html-max-z", type=int, default=256,
                   help="Max. z-Ebenen im eingebetteten Volumen "
                        "(hoeher = schaerfer in der Hoehe, groessere Datei).")
    p.add_argument("--no-mesh", action="store_true", help="Kein STL-Mesh erzeugen.")
    p.add_argument("--no-html", action="store_true", help="Keinen HTML-Schnitt-Viewer schreiben.")
    p.add_argument("--no-gui", action="store_true", help="Keinen GUI-Dialog verwenden.")
    args = p.parse_args(argv)

    folder = args.input
    if folder is None and not args.no_gui:
        folder = _pick_folder_gui()
    if not folder:
        print("Kein Eingabeordner angegeben/ausgewaehlt.", file=sys.stderr)
        return 2
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"Ordner existiert nicht: {folder}", file=sys.stderr)
        return 2

    files = _list_image_files(folder, recursive=bool(args.recursive))
    if not files:
        print(f"Keine Bilder in {folder} gefunden "
              f"(evtl. --recursive verwenden).", file=sys.stderr)
        return 1
    paths, depths = _sorted_slices(files)
    paths, depths = _subsample(paths, depths, int(args.max_slices))
    print(f"Gefundene Slices: {len(paths)}  (Tiefe {depths.min():.3f} .. {depths.max():.3f} mm)")

    vol_raw, factor = _load_volume(paths, int(args.max_inplane))
    print(f"Volumen (roh): {vol_raw.shape}  (D, H, W)")

    # In-Plane-Voxelgroesse (mm) aus dem Format.
    voxel_um, vox_src = _infer_voxel_um(folder, args.voxel_um)
    sxy_mm = (voxel_um / 1000.0) * factor

    # Geometrie-Volumen: Slices an ihre ECHTEN Tiefen setzen (korrigiert den
    # Aspekt bei ungleichmaessiger Abtastung). Fallback: gleichabstaendig stapeln.
    use_true = (not args.no_true_depth) and depths.size >= 2 and float(depths.max()) > float(depths.min())
    if use_true:
        vol_geo, sz_mm, z0_mm = _resample_true_depth(vol_raw, depths, sxy_mm, max_z=384)
        z_src = f"echte Tiefen -> {vol_geo.shape[0]} Ebenen"
    else:
        if args.z_um is not None:
            sz_mm = float(args.z_um) / 1000.0
            z_src = "override"
        else:
            step = _z_step_mm(depths)
            sz_mm = step if step is not None else sxy_mm
            z_src = "median abs_depth" if step is not None else "Fallback = in-plane"
        vol_geo = vol_raw
        z0_mm = float(depths.min()) if depths.size else 0.0

    spacing_geo = (sz_mm, sxy_mm, sxy_mm)
    stretch = sz_mm / sxy_mm if sxy_mm > 0 else 1.0
    print(f"In-Plane-Voxel: {sxy_mm * 1000:.2f} um ({vox_src}"
          f"{', downsample x%.2f' % factor if factor > 1 else ''})")
    print(f"z-Abstand     : {sz_mm * 1000:.2f} um ({z_src})")
    print(f"Geometrie-Volumen: {vol_geo.shape}  | z/xy = {stretch:.2f}")

    out = args.output or os.path.join(os.path.dirname(folder),
                                      os.path.basename(folder) + "_3d")
    os.makedirs(out, exist_ok=True)

    # 1) Rohvolumen (unveraenderte Slices, wie aufgenommen)
    _save_tiff(os.path.join(out, "volume.tif"), vol_raw)
    np.save(os.path.join(out, "volume.npy"), vol_raw)
    print("  geschrieben: volume.tif / volume.npy")

    # 2) Orthogonale Vorschau (aspekt-korrekt)
    try:
        save_orthoviews(os.path.join(out, "preview_ortho.png"), vol_geo, spacing_geo)
        print("  geschrieben: preview_ortho.png")
    except Exception as e:
        print(f"  [Vorschau] fehlgeschlagen: {e}")

    # 3) Interaktiver Viewer (HTML): Volume-Rendering (3D-Schnitt) + 2D-Schnitte
    if not args.no_html:
        vol_h, sp_h = _downsample_vol(vol_geo, spacing_geo, int(args.html_max_inplane), int(args.html_max_z))
        write_html_viewer(os.path.join(out, "view_3d.html"), vol_h, sp_h, z0_mm,
                          title=os.path.basename(folder))
        mb = os.path.getsize(os.path.join(out, "view_3d.html")) / 1e6
        print(f"  geschrieben: view_3d.html  (Volumen {vol_h.shape}, {mb:.1f} MB)")

    # 4) Mesh / STL (fuer Blender etc.; aus groeberem Volumen -> handhabbare Datei)
    mesh_info = None
    if not args.no_mesh:
        vol_m, sp_m = _downsample_vol(vol_geo, spacing_geo, 128, 256)
        res = build_mesh(vol_m, sp_m, args.iso_level)
        if res is not None:
            verts, faces, level = res
            write_binary_stl(os.path.join(out, "surface.stl"), verts, faces)
            mesh_info = {"iso_level": level, "n_vertices": int(verts.shape[0]),
                         "n_faces": int(faces.shape[0])}
            print(f"  geschrieben: surface.stl  (Level {level:.1f}, {faces.shape[0]} Dreiecke)")
            if args.render_3d:
                try:
                    render_3d(os.path.join(out, "preview_3d.png"), verts, faces)
                    print("  geschrieben: preview_3d.png")
                except Exception as e:
                    print(f"  [3D-Render] fehlgeschlagen: {e}")

    # 5) Aspekt-korrektes Volumen als TIFF
    if args.isotropic:
        try:
            if use_true:
                _save_tiff(os.path.join(out, "volume_geo.tif"), vol_geo)
                print(f"  geschrieben: volume_geo.tif  {vol_geo.shape}")
            else:
                vol_iso = make_isotropic(vol_raw, spacing_geo)
                _save_tiff(os.path.join(out, "volume_isotropic.tif"), vol_iso)
                print(f"  geschrieben: volume_isotropic.tif  {vol_iso.shape}")
        except Exception as e:
            print(f"  [isotrop] fehlgeschlagen: {e}")

    # 6) Metadaten
    meta = {
        "input_folder": folder,
        "n_slices": len(paths),
        "volume_raw_shape_DHW": list(vol_raw.shape),
        "geometry_shape_DHW": list(vol_geo.shape),
        "depth_min_mm": float(depths.min()),
        "depth_max_mm": float(depths.max()),
        "voxel_inplane_um": sxy_mm * 1000.0,
        "voxel_inplane_source": vox_src,
        "z_spacing_um": sz_mm * 1000.0,
        "z_spacing_source": z_src,
        "depth_stretch_z_over_xy": stretch,
        "inplane_downsample_factor": factor,
        "true_depth_resampled": bool(use_true),
        "mesh": mesh_info,
    }
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nFertig. Ausgabeordner:\n  {out}")
    print("Ansehen: view_3d.html per Doppelklick -> Ebene waehlen, Schieber bewegen (2D-Schnitt).")
    print("         volume.tif in Fiji/ImageJ; surface.stl in Blender/Windows 3D-Viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
