"""COCO -> YOLO + uji bolak-balik.
Pakai: python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
Jika dataset_yolo/labels ada di sebelah json, hasil dibandingkan dengan label asli."""
import sys, json
from pathlib import Path

def baca(path):
    return sorted(tuple(map(float, l.split())) for l in Path(path).read_text().splitlines() if l.strip())

def main(js, out):
    js = Path(js); d = json.loads(js.read_text())
    out = Path(out) / "labels"; out.mkdir(parents=True, exist_ok=True)
    img = {i["id"]: i for i in d["images"]}; per = {i: [] for i in img}
    for a in d["annotations"]:
        i = img[a["image_id"]]; x, y, w, h = a["bbox"]
        per[a["image_id"]].append(f"{a['category_id']-1} {(x+w/2)/i['width']:.6f} {(y+h/2)/i['height']:.6f} {w/i['width']:.6f} {h/i['height']:.6f}")
    for iid, ls in per.items():
        (out / (Path(img[iid]["file_name"]).stem + ".txt")).write_text("\n".join(ls) + ("\n" if ls else ""))
    print(f"{len(per)} file label ditulis ke {out}")
    asli = js.parent / "labels"
    if not asli.exists(): return
    bad = 0; maxd = 0.0
    for t in asli.glob("*.txt"):
        b = out / t.name
        if not b.exists(): bad += 1; print("HILANG", t.name); continue
        A, B = baca(t), baca(b)
        if len(A) != len(B): bad += 1; print("JUMLAH BEDA", t.name); continue
        for r, s in zip(A, B):
            if r[0] != s[0]: bad += 1; print("KELAS BEDA", t.name); break
            dd = max(abs(u - v) for u, v in zip(r[1:], s[1:])); maxd = max(maxd, dd)
            if dd > 1e-4: bad += 1; print("KOTAK BEDA", t.name, f"{dd:.6f}"); break
    print(f"Uji bolak-balik: {'LOLOS' if bad == 0 else 'GAGAL'} (selisih maks {maxd:.2e}, {bad} berkas bermasalah)")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
