"""Pengganti 00_ambil_subset_coco.py untuk dataset sendiri (tanpa COCO).
Memakai folder per-kelas (dataset/botol, dataset/box), membuang gambar yang
hampir identik, lalu menyalin ke dataset_yolo/images + classes.txt.
Label (.txt YOLO) dibuat lewat anotasi manual (LabelImg/CVAT/Roboflow).

Pakai: python 00_siapkan_data.py dataset dataset_yolo [--ambang 3]
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

EXT = {".jpg", ".jpeg", ".png"}

def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=float).ravel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sumber"); ap.add_argument("keluar")
    ap.add_argument("--ambang", type=float, default=3.0,
                    help="jarak RMS thumbnail 16x16; di bawah ini dianggap kembar")
    a = ap.parse_args()
    src, out = Path(a.sumber), Path(a.keluar)
    kelas = sorted(d.name for d in src.iterdir()
                   if d.is_dir() and d.name not in {"train", "val", "test"})
    if not kelas: sys.exit("Tidak ada folder kelas di " + str(src))
    img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "classes.txt").write_text("\n".join(kelas) + "\n")
    print("Kelas (urutan = nomor kelas YOLO):", {i: k for i, k in enumerate(kelas)})
    total = 0
    for k in kelas:
        files = sorted(p for p in (src / k).iterdir() if p.suffix.lower() in EXT)
        kept, T = [], []
        for p in files:
            t = thumb(p)
            if all(((t - u) ** 2).mean() ** .5 >= a.ambang for u in T):
                kept.append(p); T.append(t)
        for p in kept:
            shutil.copy2(p, img_dir / f"{k}_{p.name}")
        print(f"{k:8s}: {len(files)} gambar -> {len(kept)} dipakai ({len(files)-len(kept)} kembar dibuang)")
        total += len(kept)
    print(f"Total {total} gambar di {img_dir}")
    print("Langkah berikut: anotasi semua gambar, simpan .txt YOLO ke", out / "labels")

if __name__ == "__main__":
    main()
