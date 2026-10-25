"""Cek format label YOLO.  Pakai: python 01_cek_anotasi.py dataset_yolo"""
import sys, collections
from pathlib import Path
from PIL import Image

EXT = {".jpg", ".jpeg", ".png"}

def main(root):
    root = Path(root)
    kelas = [l.strip() for l in (root / "classes.txt").read_text().splitlines() if l.strip()]
    imgs = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in EXT)
    lab = root / "labels"
    err, warn = [], []
    cnt = collections.Counter(); n_kosong = 0
    for p in imgs:
        t = lab / (p.stem + ".txt")
        if not t.exists():
            err.append(f"{p.name}: file label tidak ada"); continue
        lines = [l for l in t.read_text().splitlines() if l.strip()]
        if not lines:
            n_kosong += 1; warn.append(f"{t.name}: kosong (gambar tanpa objek)"); continue
        seen = set()
        for i, l in enumerate(lines, 1):
            f = l.split()
            if len(f) != 5:
                err.append(f"{t.name}:{i}: harus 5 kolom, ada {len(f)}"); continue
            try:
                c = int(f[0]); cx, cy, w, h = map(float, f[1:])
            except ValueError:
                err.append(f"{t.name}:{i}: nilai bukan angka"); continue
            if not 0 <= c < len(kelas): err.append(f"{t.name}:{i}: kelas {c} di luar 0..{len(kelas)-1}")
            if not all(0 <= v <= 1 for v in (cx, cy, w, h)): err.append(f"{t.name}:{i}: nilai di luar 0-1 (bukan format YOLO ternormalisasi?)")
            if w <= 0 or h <= 0: err.append(f"{t.name}:{i}: lebar/tinggi <= 0")
            if cx - w/2 < -1e-6 or cx + w/2 > 1 + 1e-6 or cy - h/2 < -1e-6 or cy + h/2 > 1 + 1e-6:
                err.append(f"{t.name}:{i}: kotak keluar dari gambar")
            if w * h < 0.0005: warn.append(f"{t.name}:{i}: kotak sangat kecil")
            if l in seen: err.append(f"{t.name}:{i}: baris duplikat")
            seen.add(l)
            if 0 <= c < len(kelas): cnt[kelas[c]] += 1
    yatim = [t.name for t in lab.glob("*.txt") if not any(p.stem == t.stem for p in imgs)]
    for y in yatim: err.append(f"{y}: label tanpa gambar")
    print(f"Gambar: {len(imgs)} | label kosong: {n_kosong}")
    print("Kotak per kelas:", dict(cnt))
    for w in warn[:20]: print("PERINGATAN", w)
    for e in err[:50]: print("ERROR", e)
    print("HASIL:", "GAGAL" if err else "LOLOS", f"({len(err)} error, {len(warn)} peringatan)")
    sys.exit(1 if err else 0)

if __name__ == "__main__":
    main(sys.argv[1])
