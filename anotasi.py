"""Alat anotasi bounding box sederhana (OpenCV) -> label YOLO.
Pasang:  pip install opencv-python
Pakai :  python anotasi.py dataset_yolo

Mouse : tarik tombol kiri untuk menggambar kotak
Tombol: 0-9  pilih kelas (default otomatis dari awalan nama file, mis. botol_xxx.jpg)
        u    batalkan kotak terakhir      c   hapus semua kotak gambar ini
        n/spasi/Enter  simpan lalu gambar berikut      b  simpan lalu gambar sebelumnya
        q/Esc  simpan lalu keluar
Gambar tanpa objek: tekan n tanpa menggambar -> dianggap kosong (file .txt kosong).
"""
import sys
from pathlib import Path
import cv2

EXT = {".jpg", ".jpeg", ".png"}
WARNA = [(0, 200, 0), (0, 140, 255), (255, 80, 80), (200, 0, 200), (0, 220, 220)]
LAYAR_MAKS = (1000, 700)  # lebar, tinggi jendela maksimum

def ke_yolo(x1, y1, x2, y2, W, H):
    """Koordinat piksel gambar asli -> (cx, cy, w, h) ternormalisasi 0-1."""
    x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
    y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)

def dari_yolo(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)

def baca_label(path, W, H):
    kotak = []
    if path.exists():
        for l in path.read_text().splitlines():
            f = l.split()
            if len(f) == 5:
                c = int(f[0]); cx, cy, w, h = map(float, f[1:])
                kotak.append((c, *dari_yolo(cx, cy, w, h, W, H)))
    return kotak

def simpan_label(path, kotak, W, H):
    baris = []
    for c, x1, y1, x2, y2 in kotak:
        cx, cy, w, h = ke_yolo(x1, y1, x2, y2, W, H)
        baris.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(baris) + ("\n" if baris else ""))

def main(root):
    root = Path(root)
    kelas = [l.strip() for l in (root / "classes.txt").read_text().splitlines() if l.strip()]
    imgs = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in EXT)
    lab = root / "labels"; lab.mkdir(exist_ok=True)
    idx = next((i for i, p in enumerate(imgs) if not (lab / (p.stem + ".txt")).exists()), 0)
    st = {"drag": None, "cur": None, "kotak": [], "skala": 1.0, "kelas": 0}
    win = "anotasi"

    def mouse(ev, x, y, fl, _):
        s = st["skala"]
        if ev == cv2.EVENT_LBUTTONDOWN: st["drag"] = (x, y); st["cur"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]: st["cur"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
            x0, y0 = st["drag"]; st["drag"] = None
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                st["kotak"].append((st["kelas"], x0 / s, y0 / s, x / s, y / s))

    cv2.namedWindow(win); cv2.setMouseCallback(win, mouse)
    while 0 <= idx < len(imgs):
        p = imgs[idx]; im = cv2.imread(str(p)); H, W = im.shape[:2]
        st["skala"] = min(LAYAR_MAKS[0] / W, LAYAR_MAKS[1] / H)
        s = st["skala"]; dasar = cv2.resize(im, (int(W * s), int(H * s)))
        st["kotak"] = baca_label(lab / (p.stem + ".txt"), W, H)
        st["kelas"] = next((i for i, k in enumerate(kelas) if p.name.startswith(k + "_")), 0)
        arah = 0
        while arah == 0:
            v = dasar.copy()
            for c, x1, y1, x2, y2 in st["kotak"]:
                w_ = WARNA[c % len(WARNA)]
                cv2.rectangle(v, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), w_, 2)
                cv2.putText(v, kelas[c], (int(x1 * s) + 3, int(y1 * s) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, w_, 2)
            if st["drag"]: cv2.rectangle(v, st["drag"], st["cur"], WARNA[st["kelas"] % len(WARNA)], 1)
            info = f"[{idx+1}/{len(imgs)}] {p.name} | kelas aktif: {st['kelas']}={kelas[st['kelas']]} | kotak: {len(st['kotak'])}"
            cv2.rectangle(v, (0, 0), (v.shape[1], 22), (30, 30, 30), -1)
            cv2.putText(v, info, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, v); k = cv2.waitKey(20) & 0xFF
            if k in (ord("n"), 32, 13): arah = 1
            elif k == ord("b"): arah = -1
            elif k in (ord("q"), 27): arah = 99
            elif k == ord("u") and st["kotak"]: st["kotak"].pop()
            elif k == ord("c"): st["kotak"] = []
            elif ord("0") <= k <= ord("9") and int(chr(k)) < len(kelas): st["kelas"] = int(chr(k))
        simpan_label(lab / (p.stem + ".txt"), st["kotak"], W, H)
        if arah == 99: break
        idx += arah
    cv2.destroyAllWindows()
    n = sum((lab / (p.stem + ".txt")).exists() for p in imgs)
    print(f"Tersimpan: {n}/{len(imgs)} gambar berlabel. Lanjut: python 01_cek_anotasi.py {root}")

if __name__ == "__main__":
    main(sys.argv[1])
