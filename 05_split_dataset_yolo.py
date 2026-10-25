"""Bagi dataset YOLO menjadi train/val/test TANPA kebocoran antar-bingkai mirip.
Gambar berurutan (urut nama file) yang mirip dirantai jadi satu kelompok; satu
kelompok tidak pernah dipecah ke dua bagian. Pembagian per kelas (stratified).

Pakai: python 05_split_dataset_yolo.py dataset_yolo dataset_v1 [--rasio 0.7 0.2 0.1] [--seed 42] [--ambang 6]
"""
import argparse, random, shutil, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image

EXT = {".jpg", ".jpeg", ".png"}
BAGIAN = ["train", "val", "test"]

def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=float).ravel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sumber"); ap.add_argument("tujuan")
    ap.add_argument("--rasio", type=float, nargs=3, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ambang", type=float, default=6.0, help="jarak RMS di bawah ini = satu kelompok (bingkai berurutan)")
    a = ap.parse_args()
    src, dst = Path(a.sumber), Path(a.tujuan)
    kelas = [l.strip() for l in (src / "classes.txt").read_text().splitlines() if l.strip()]
    imgs = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in EXT)
    # kelas gambar = kelas dominan pada labelnya (untuk stratifikasi)
    def kls(p):
        t = src / "labels" / (p.stem + ".txt")
        c = [int(l.split()[0]) for l in t.read_text().splitlines() if l.strip()] if t.exists() else []
        return collections.Counter(c).most_common(1)[0][0] if c else -1
    per = collections.defaultdict(list)
    for p in imgs: per[kls(p)].append(p)
    rng = random.Random(a.seed)
    tetapkan = {}
    for c, ps in sorted(per.items()):
        grup, T = [], [thumb(p) for p in ps]
        for i, p in enumerate(ps):                       # rantai bingkai berurutan yang mirip
            if grup and ((T[i] - T[i-1]) ** 2).mean() ** .5 < a.ambang: grup[-1].append(p)
            else: grup.append([p])
        rng.shuffle(grup)
        n = len(ps); kuota = [r * n for r in a.rasio]; isi = [0, 0, 0]
        grup.sort(key=len, reverse=True)                 # kelompok besar dulu
        for g in grup:
            k = max(range(3), key=lambda j: kuota[j] - isi[j])
            for p in g: tetapkan[p] = BAGIAN[k]
            isi[k] += len(g)
        nm = kelas[c] if c >= 0 else "tanpa-objek"
        print(f"{nm:8s}: {n} gambar, {len(grup)} kelompok -> train {isi[0]} / val {isi[1]} / test {isi[2]}")
    for b in BAGIAN:
        (dst / "images" / b).mkdir(parents=True, exist_ok=True); (dst / "labels" / b).mkdir(parents=True, exist_ok=True)
    for p, b in tetapkan.items():
        shutil.copy2(p, dst / "images" / b / p.name)
        t = src / "labels" / (p.stem + ".txt")
        (dst / "labels" / b / (p.stem + ".txt")).write_text(t.read_text() if t.exists() else "")
    yaml = "train: images/train\nval: images/val\ntest: images/test\n\nnc: %d\nnames:\n" % len(kelas)
    yaml += "".join(f"  {i}: {k}\n" for i, k in enumerate(kelas))
    (dst / "data.yaml").write_text(yaml)
    tot = collections.Counter(tetapkan.values())
    print("Total:", dict(tot), "-> data.yaml ditulis di", dst)
    if any(tot[b] == 0 for b in BAGIAN): print("PERINGATAN: ada bagian kosong; tambah data atau ubah --ambang")

if __name__ == "__main__":
    main()
