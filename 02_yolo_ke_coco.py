"""YOLO -> COCO.  Pakai: python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json"""
import sys, json
from pathlib import Path
from PIL import Image

EXT = {".jpg", ".jpeg", ".png"}

def main(root, out):
    root = Path(root)
    kelas = [l.strip() for l in (root / "classes.txt").read_text().splitlines() if l.strip()]
    d = {"images": [], "annotations": [],
         "categories": [{"id": i + 1, "name": k} for i, k in enumerate(kelas)]}  # COCO mulai 1
    aid = 1
    for iid, p in enumerate(sorted(q for q in (root / "images").iterdir() if q.suffix.lower() in EXT), 1):
        W, H = Image.open(p).size
        d["images"].append({"id": iid, "file_name": p.name, "width": W, "height": H})
        t = root / "labels" / (p.stem + ".txt")
        if not t.exists(): continue
        for l in t.read_text().splitlines():
            if not l.strip(): continue
            c, cx, cy, w, h = l.split(); c = int(c); cx, cy, w, h = map(float, (cx, cy, w, h))
            bw, bh = w * W, h * H
            x, y = (cx - w / 2) * W, (cy - h / 2) * H          # pojok kiri atas, piksel
            d["annotations"].append({"id": aid, "image_id": iid, "category_id": c + 1,
                "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3), "iscrowd": 0}); aid += 1
    Path(out).write_text(json.dumps(d, indent=1))
    print(f"{len(d['images'])} gambar, {len(d['annotations'])} anotasi -> {out}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
