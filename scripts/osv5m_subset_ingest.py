#!/usr/bin/env python3
"""
OSV-5M street-level subset ingest (HF-shard based).

Builds a real, region-scoped street-level reference layer for GeoVision's
retrieval index from the OpenStreetView-5M benchmark (CC-BY-SA-4.0, legit license,
Mapillary-derived street-view images with real GPS).

IMPORTANT (why shards, not the CSV thumb URLs): OSV-5M's test.csv `thumb_original_url`
fields point to a Facebook CDN that returns HTTP 403 to anonymous fetch (dead links).
The REAL images live in HF `.zip` shards (e.g. images/test/00.zip, ~2.25GB each),
which download fine from HF Hub. So this ingestor:
  1. streams test.csv, keeps rows for a target country/city
  2. downloads ONE HF shard zip per run (bounded by --max-bytes / --max-images)
  3. extracts ONLY the members whose id matches the filtered rows
  4. CLIP-embeds them and appends to data/visual_geo_db (same atomic path).

REQUIRES DISK+BANDWIDTH: each shard is ~2.25GB. Use a small --country with few rows,
or accept pulling a whole shard. This is a bounded, on-demand operation — not run
by default.

Run (background, CPU embedding is slow):
  python3 scripts/osv5m_subset_ingest.py --country FR --shard 00 --max-images 25
"""
import argparse
import csv
import io
import os
import sys
import time
import zipfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

TEST_CSV = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/test.csv"
SHARD_BASE = ("https://huggingface.co/datasets/osv5m/osv5m/resolve/main/images/test/"
              "{shard}.zip")
UA = "Mozilla/5.0 (X11; Linux x86_64) GeoVision-OSV5M/1.0"


def stream_rows():
    req = urllib.request.Request(TEST_CSV, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        reader = csv.DictReader(io.TextIOWrapper(resp, encoding="utf-8", errors="ignore"))
        yield from reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", type=str, default=None, help="ISO country code (e.g. FFIN,US,BR)")
    ap.add_argument("--city", type=str, default=None, help="city substring (case-insensitive)")
    ap.add_argument("--shard", type=str, default="00", help="HF shard number (00..NN)")
    ap.add_argument("--max-images", type=int, default=25)
    ap.add_argument("--limit-scan", type=int, default=2_000_000)
    ap.add_argument("--temp-dir", type=str, default="/tmp/osv5m_shard",
                    help="cache dir for the downloaded shard zip + extract")
    args = ap.parse_args()
    if not (args.country or args.city):
        print("need --country or --city")
        return 1

    from modules.geo_harness import GeoVisionHarness
    from modules.visual_geo_engine import VisualGeoEngine, DB_EMB, DB_META, CLIP_CACHE
    import gzip, json, requests
    import open_clip, torch
    import numpy as np

    engine = VisualGeoEngine()
    if not engine._ensure_db():
        print("visual_geo_db missing; build first")
        return 1
    existing = {(m.get("source"), m.get("title")) for m in engine._db_meta if m.get("title")}
    emb_list = [engine._db_embeddings]
    meta_list = list(engine._db_meta)

    # 1) collect matching metadata (country/city filtered)
    country_l = (args.country or "").upper()
    city_l = (args.city or "").lower()
    wanted = {}  # id -> row
    scanned = 0
    for row in stream_rows():
        scanned += 1
        if scanned > args.limit_scan:
            break
        if country_l and row.get("country", "").upper() != country_l:
            continue
        if city_l and city_l not in row.get("city", "").lower():
            continue
        wanted[row.get("id")] = row
    print(f"metadata: scanned {scanned}, matched {len(wanted)} rows for "
          f"country={country_l or 'any'} city={args.city or 'any'}")
    if not wanted:
        print("no matching rows; adjust filters")
        return 0

    # 2) download the shard zip (bounded-ish; will print size)
    shard_url = SHARD_BASE.format(shard=args.shard)
    zip_path = os.path.join(args.temp_dir, f"{args.shard}.zip")
    os.makedirs(args.temp_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"downloading {shard_url} -> {zip_path} (could be ~2.25GB)")
        req = urllib.request.Request(shard_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as resp, open(zip_path, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    else:
        print(f"shard already present: {zip_path}")

    # 3) extract ONLY members whose filename (id.jpg) is in `wanted`
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(CLIP_CACHE))
    model.eval().to(device)

    added = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"shard has {len(names)} files; extracting matching ids...")
        for name in names:
            if added >= args.max_images:
                break
            base = os.path.basename(name)
            oid = os.path.splitext(base)[0]
            row = wanted.get(oid)
            if row is None:
                continue
            key = ("osv5m", oid)
            if key in existing:
                continue
            try:
                data = zf.read(name)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                tens = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(tens)
                emb = emb.squeeze(0).float().cpu().numpy()
                norm = np.linalg.norm(emb)
                if norm == 0:
                    skipped += 1
                    continue
                emb_list.append((emb / norm).reshape(1, -1).astype(np.float32))
                meta_list.append({
                    "lat": float(row["latitude"]), "lon": float(row["longitude"]),
                    "title": oid, "city": row.get("city", ""),
                    "country": row.get("country", ""), "population": 0,
                    "source": "osv5m",
                })
                existing.add(key)
                added += 1
                print(f"[+{added}] {row.get('country')}/{row.get('city','')} "
                      f"@ {float(row['latitude']):.4f},{float(row['longitude']):.4f}")
            except Exception:
                skipped += 1
                continue

    if added == 0:
        print(f"no OSV-5M images appended from shard {args.shard} (skipped {skipped}). "
              "Id-matching may need the same shard the target country's images live in.")
        return 0

    arr = np.concatenate([a for a in emb_list if a is not None], axis=0).astype(np.float32)
    tmp = DB_EMB.with_suffix(".tmp.npy")
    np.save(tmp, arr); tmp.replace(DB_EMB)
    with gzip.open(DB_META, "wt", encoding="utf-8") as f:
        for m in meta_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"OSV-5M ingest: +{added} real street photos (skipped {skipped}). "
          f"DB now {int(arr.shape[0])} refs. Delete osv5m_shard dir to reclaim disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())