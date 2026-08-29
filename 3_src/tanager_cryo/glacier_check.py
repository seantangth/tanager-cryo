"""Audit every scene of the Tanager Open STAC catalogue against mapped glaciers.

The memo's title claim -- the open hyperspectral archive contains no glacier -- is
empirical, so it is computed here rather than asserted. Every unique scene in the
catalogue is queried against the OpenStreetMap ``natural=glacier`` layer via the
Overpass API, using the scene's own STAC bounding box. A bounding box overstates a
rotated footprint, which for this test is the conservative direction: it can only
create intersections, never hide one.

OpenStreetMap is not a glacier inventory; its Arctic coverage leans on national
imports (e.g. CanVec for Canada) and is incomplete in places. A zero here is
therefore a strong indication, not a proof, and the scene this submission analyses
was additionally checked by inspection.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.glacier_check --out 5_outputs/glacier_audit.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .fetch import CATALOG, COLLECTIONS, _get_json

ROOT = Path(__file__).resolve().parents[2]
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "tanager-cryo/0.1 (Tanager Open Data Competition)"


def glacier_count(bbox: list[float], timeout: int = 90) -> int | None:
    """Number of OSM glacier ways/relations intersecting a lon/lat bbox.

    Returns None if Overpass declines the request; the caller records the scene as
    skipped rather than failing the whole audit.
    """
    q = (
        f'[out:json][timeout:60];('
        f'way["natural"="glacier"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
        f'relation["natural"="glacier"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}););'
        f"out count;"
    )
    req = urllib.request.Request(
        OVERPASS,
        data=urllib.parse.urlencode({"data": q}).encode(),
        headers={"User-Agent": USER_AGENT},
    )
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as exc:
        print(f"    Overpass unavailable ({exc}); skipping")
        return None
    return int(d["elements"][0]["tags"]["total"]) if d.get("elements") else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs" / "glacier_audit.json")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between Overpass requests, to stay under rate limits")
    args = ap.parse_args()

    # Collect every unique scene. Scenes appear in more than one collection, so the
    # item is fetched once per scene id.
    items: dict[str, list[float]] = {}
    per_collection: dict[str, int] = {}
    for col in COLLECTIONS:
        try:
            c = _get_json(f"{CATALOG}/{col}/collection.json")
        except Exception as exc:
            print(f"collection {col}: unavailable ({exc})")
            continue
        ids = [
            link["href"].rsplit("/", 1)[-1].removesuffix(".json")
            for link in c["links"]
            if link.get("rel") == "item"
        ]
        per_collection[col] = len(ids)
        for sid in ids:
            if sid in items:
                continue
            try:
                item = _get_json(f"{CATALOG}/{col}/{sid}/{sid}.json")
            except Exception as exc:
                print(f"  {sid}: item fetch failed ({exc})")
                continue
            items[sid] = item["bbox"]
        print(f"collection {col}: {len(ids)} items ({len(items)} unique scenes so far)")

    results: dict[str, int | None] = {}
    hits = []
    for i, (sid, bbox) in enumerate(sorted(items.items()), 1):
        n = glacier_count(bbox)
        results[sid] = n
        if n:
            hits.append(sid)
            print(f"  [{i}/{len(items)}] {sid}  glaciers={n}  <-- INTERSECTION")
        elif n == 0:
            print(f"  [{i}/{len(items)}] {sid}  glaciers=0")
        time.sleep(args.sleep)

    checked = sum(1 for v in results.values() if v is not None)
    skipped = sum(1 for v in results.values() if v is None)
    print(
        f"\n{len(items)} unique scenes; {checked} checked, {skipped} skipped; "
        f"{len(hits)} intersect a mapped glacier"
    )

    payload = {
        "date": dt.date.today().isoformat(),
        "catalog": CATALOG,
        "items_per_collection": per_collection,
        "n_unique_scenes": len(items),
        "n_checked": checked,
        "n_skipped": skipped,
        "n_intersecting": len(hits),
        "intersecting_scenes": hits,
        "glacier_counts": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
