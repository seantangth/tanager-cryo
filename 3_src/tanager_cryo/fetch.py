"""Download a Tanager scene from Planet's public Open STAC catalogue.

No authentication is required: the assets sit on public Google Cloud Storage under a
CC-BY 4.0 licence. This module resolves a scene id to its ortho surface-reflectance asset
by walking the catalogue, so a fresh clone can reproduce the analysis end to end.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.fetch --scene 20250606_181248_58_4001
    PYTHONPATH=3_src python -m tanager_cryo.fetch --list-collection snow-ice
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = "https://www.planet.com/data/stac/tanager-core-imagery"
COLLECTIONS = (
    "snow-ice",
    "natural-lands",
    "coastal-water-bodies",
    "fire",
    "urban",
    "agriculture",
    "energy-mining",
    "GHG-plumes",
    "ROCX2025",
)

# Where each scene lands. Keyed by scene id so a reader can see, from this file alone,
# which scene the analysis in the memo used.
DEFAULT_DESTINATIONS = {
    "20250606_181248_58_4001": ROOT / "1_data" / "raw" / "sirmilik",
}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as fh:
        return json.load(fh)


def find_item(scene_id: str) -> dict:
    """Locate a scene's STAC item by trying each collection in turn."""
    for col in COLLECTIONS:
        try:
            return _get_json(f"{CATALOG}/{col}/{scene_id}/{scene_id}.json")
        except Exception:
            continue
    raise LookupError(f"scene {scene_id} not found in any collection of {CATALOG}")


def download_asset(item: dict, asset: str, dest_dir: Path, suffix: str) -> Path:
    href = item["assets"][asset]["href"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{item['id']}{suffix}"
    if out.exists():
        print(f"  {out.name} already present ({out.stat().st_size / 1e6:.0f} MB), skipping")
        return out
    print(f"  downloading {asset} -> {out.name}")
    with urllib.request.urlopen(href, timeout=1800) as r, open(out, "wb") as fh:
        shutil.copyfileobj(r, fh)
    print(f"  done ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", help="Tanager scene id, e.g. 20250606_181248_58_4001")
    ap.add_argument("--dest", type=Path, help="destination directory")
    ap.add_argument(
        "--assets",
        nargs="+",
        default=["ortho_sr_hdf5", "ortho_visual"],
        help="STAC asset keys to fetch",
    )
    ap.add_argument("--list-collection", help="list the scene ids in one collection")
    args = ap.parse_args()

    if args.list_collection:
        col = _get_json(f"{CATALOG}/{args.list_collection}/collection.json")
        items = [l["href"].rsplit("/", 1)[-1].removesuffix(".json")
                 for l in col.get("links", []) if l["rel"] == "item"]
        print(f"{args.list_collection}: {len(items)} scenes")
        for i in items:
            print(f"  {i}")
        return

    if not args.scene:
        ap.error("--scene is required unless --list-collection is given")

    item = find_item(args.scene)
    dest = args.dest or DEFAULT_DESTINATIONS.get(args.scene, ROOT / "1_data" / "raw" / args.scene)
    try:
        shown = dest.resolve().relative_to(ROOT)
    except ValueError:
        # A --dest outside the repository is perfectly legitimate; show it absolute.
        shown = dest
    print(f"{args.scene} -> {shown}")
    suffixes = {"ortho_sr_hdf5": "_sr.h5", "ortho_visual": "_visual.tif",
                "ortho_radiance_hdf5": "_rad.h5", "thumbnail": "_thumb.png"}
    for a in args.assets:
        if a not in item["assets"]:
            print(f"  [skip] no asset '{a}' on this item")
            continue
        download_asset(item, a, dest, suffixes.get(a, f"_{a}"))


if __name__ == "__main__":
    main()
