"""Tahap 3-5: latih E1 (pretrained), E2 (dari nol), E3 (satu perubahan), pilih model
terbaik berdasarkan mAP50-95 val, uji SEKALI di data test.
Padanan buatan sendiri untuk 06_latih_bandingkan.py di slide 35-36 (bukan skrip dosen).

Pakai (Colab/GPU):
  python 06_latih_bandingkan.py --nim 4222401070 --nama "Nama Anda"
  python 06_latih_bandingkan.py --nim ... --nama "..." --ubah imgsz=640
  python 06_latih_bandingkan.py --nim ... --nama "..." --ubah model=yolov8s.pt     # ganti model, bukan imgsz
Keluaran: hasil_praktik/E1_pretrained, E2_scratch, E3_tuning, serta
          hasil_praktik_<NIM>.txt dan .csv
"""
import argparse, csv, hashlib, platform, sys, time
from datetime import datetime
from pathlib import Path


def sidik_jari(data_yaml):
    """SHA-256 (12 hex pertama) dari seluruh label dataset: nama + isi, urut."""
    root = Path(data_yaml).parent
    h = hashlib.sha256()
    for t in sorted((root / "labels").rglob("*.txt")):
        h.update(str(t.relative_to(root)).encode()); h.update(t.read_bytes())
    return h.hexdigest()[:12]


def parse_ubah(daftar):
    """['imgsz=640', 'model=yolov8s.pt'] -> {'imgsz': 640, 'model': 'yolov8s.pt'}"""
    d = {}
    for s in daftar:
        k, v = s.split("=", 1)
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError: pass
        d[k] = v
    return d


def baris_tabel(nama, ubah, r):
    return (f"  {nama:14s} {ubah:14s} {r['map50']:.3f}  {r['map5095']:.3f}  "
            f"{r['p']:.3f} {r['r']:.3f} {r['ms']:5.1f} {r['menit']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nim", required=True); ap.add_argument("--nama", required=True)
    ap.add_argument("--data", default="dataset_v1/data.yaml")
    ap.add_argument("--model", default="yolov8n", help="nama dasar tanpa ekstensi, mis. yolov8n")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proyek", default="hasil_praktik")
    ap.add_argument("--ubah", nargs="+", default=["imgsz=640"],
                    help="satu perubahan untuk E3, format kunci=nilai (default imgsz=640)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO  # impor di sini agar fungsi lain bisa dipakai tanpa ultralytics
    data = str(Path(a.data).resolve())          # harus absolut agar data.yaml terbaca benar
    proyek = str(Path(a.proyek).resolve())
    sj = sidik_jari(data)
    ubah = parse_ubah(a.ubah)
    ubah_str = " ".join(a.ubah)
    umum = dict(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
                project=proyek, exist_ok=True, plots=True, verbose=False)
    if a.device is not None: umum["device"] = a.device

    rencana = [("E1_pretrained", "-", f"{a.model}.pt", {}, True),
               ("E2_scratch", "-", f"{a.model}.yaml", {"pretrained": False}, False),
               ("E3_tuning", ubah_str, ubah.pop("model", f"{a.model}.pt"), ubah, True)]
    hasil = {}
    for nama, uc, bobot, extra, _ in rencana:
        print(f"\n=== {nama}: model={bobot} {extra} ===", flush=True)
        t0 = time.time()
        m = YOLO(bobot)
        m.train(name=nama, **{**umum, **extra})
        menit = (time.time() - t0) / 60
        best = Path(proyek) / nama / "weights" / "best.pt"
        v = YOLO(str(best)).val(data=data, split="val", imgsz=extra.get("imgsz", a.imgsz),
                                project=proyek, name=nama + "_val", exist_ok=True, verbose=False, plots=False)
        hasil[nama] = dict(ubah=uc, map50=float(v.box.map50), map5095=float(v.box.map),
                           p=float(v.box.mp), r=float(v.box.mr), ms=float(v.speed["inference"]),
                           menit=menit, best=str(best), imgsz=extra.get("imgsz", a.imgsz))

    terpilih = max(hasil, key=lambda k: hasil[k]["map5095"])
    tv = YOLO(hasil[terpilih]["best"]).val(data=data, split="test", imgsz=hasil[terpilih]["imgsz"],
                                           project=proyek, name="TEST_" + terpilih, exist_ok=True,
                                           verbose=False, plots=True)
    test = dict(map50=float(tv.box.map50), map5095=float(tv.box.map), p=float(tv.box.mp), r=float(tv.box.mr))

    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baris = [f"  NIM / Nama    : {a.nim} / {a.nama}",
             f"  Waktu         : {waktu}   Komputer: {platform.node()}",
             f"  Dataset       : {Path(a.data).parent.name}  (sidik jari label: {sj})",
             f"  Model dasar   : {a.model}  epochs={a.epochs} imgsz={a.imgsz} batch={a.batch} seed={a.seed}",
             "  Eksperimen     Perubahan      mAP50 mAP50-95    P     R    ms  menit"]
    for n, r in hasil.items(): baris.append(baris_tabel(n, r["ubah"], r))
    baris.append(f"  Model terpilih (mAP50-95 val tertinggi): {terpilih}")
    baris.append(f"  Hasil di data TEST (sekali saja)       : mAP50={test['map50']:.3f}  mAP50-95={test['map5095']:.3f}  P={test['p']:.3f}  R={test['r']:.3f}")
    teks = "\n".join(baris)
    print("\n" + teks)
    Path(f"hasil_praktik_{a.nim}.txt").write_text(teks + "\n")
    with open(f"hasil_praktik_{a.nim}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nim", "nama", "waktu", "komputer", "dataset_sidik_jari", "eksperimen", "perubahan",
                    "mAP50", "mAP50-95", "P", "R", "ms", "menit"])
        for n, r in hasil.items():
            w.writerow([a.nim, a.nama, waktu, platform.node(), sj, n, r["ubah"], f"{r['map50']:.4f}",
                        f"{r['map5095']:.4f}", f"{r['p']:.4f}", f"{r['r']:.4f}", f"{r['ms']:.2f}", f"{r['menit']:.2f}"])
        w.writerow([a.nim, a.nama, waktu, platform.node(), sj, "TEST:" + terpilih, "-", f"{test['map50']:.4f}",
                    f"{test['map5095']:.4f}", f"{test['p']:.4f}", f"{test['r']:.4f}", "", ""])
    print(f"\nTersimpan: hasil_praktik_{a.nim}.txt / .csv dan folder {a.proyek}/")


if __name__ == "__main__":
    main()
