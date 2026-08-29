"""Quantify the spaceborne imaging-spectroscopy gap over cryosphere hazard terrain.

The argument the submission rests on is empirical, not rhetorical, so it is computed
here from public catalogues rather than asserted.

Two sites, two different failures of the same kind:

**Langtang, Nepal (28.2853 N, 85.5252 E).** On 2026-08-26 a combined rock and ice slope
failure on the north side of Langtang Lirung sent bedrock and glacier ice into the Lhende
Khola. Satellite imagery shows a scar with both ice *and* rock removed, so this is a
rock-ice avalanche rather than the clean ice detachment of the 2016 Aru type, and it is not
a glacial lake outburst flood. The debris temporarily dammed the river; the impoundment
breached, and the flood ran ~100 km down the Bhote Koshi and Trishuli, destroying the
Gyirong border crossing. EMIT can see this latitude. The question is how often it did.

**Sirmilik, Nunavut (73.7 N).** EMIT cannot see this latitude at all -- the ISS orbit
stops near 52 degrees -- and PRISMA acquires only within +/- 70 degrees. EnMAP's polar
orbit can reach it, but its proposal-driven tasked archive holds no systematic sea-ice
coverage. No spaceborne imaging spectrometer systematically observes the Arctic sea-ice
zone; Tanager, in a polar orbit, is positioned to.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.observation_gap --out 5_outputs/observation_gap.json
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Failure source zone. Two independent positions have been published: 85.5194, 28.2765
# (EOS Landslide Blog) and 85.52515, 28.28532 (AntarcticGlaciers.org). They differ by
# 1.1 km. We use the latter and note that the coverage result is insensitive to the
# choice -- both return 20 granules, 2 usable, and the same 28-month gap, because an EMIT
# granule is ~75 km across.
LANGTANG = (85.52515461726692, 28.28531746075177)
LANGTANG_ALT = (85.5194, 28.2765)
LANGTANG_FAILURE_DATE = "2026-08-26"

# The Tanager scene this submission analyses.
SIRMILIK = (-81.68, 73.72)

# ISS inclination limits EMIT to roughly +/- 52 degrees.
EMIT_LATITUDE_LIMIT = 52.0

# Scene-level cloud fraction above which an optical observation of a mountain headwall is
# not usable. Generous: in practice a 55%-cloud scene is often unusable over the target.
USABLE_CLOUD_MAX = 55.0


def query_point(lon: float, lat: float, short_name: str = "EMITL2ARFL") -> list[dict]:
    """All granules of ``short_name`` whose footprint contains the point."""
    q = urllib.parse.urlencode(
        {"short_name": short_name, "point": f"{lon},{lat}", "page_size": 200}
    )
    with urllib.request.urlopen(f"{CMR}?{q}", timeout=120) as fh:
        entries = json.load(fh)["feed"]["entry"]
    out = []
    for g in entries:
        cc = g.get("cloud_cover")
        out.append(
            {
                "time": g.get("time_start", "")[:10],
                "cloud_cover": float(cc) if cc not in (None, "") else None,
                "granule": g.get("producer_granule_id"),
            }
        )
    return sorted(out, key=lambda r: r["time"])


def summarise(records: list[dict], failure_date: str | None = None) -> dict:
    usable = [
        r for r in records
        if r["cloud_cover"] is not None and r["cloud_cover"] < USABLE_CLOUD_MAX
    ]
    out = {
        "n_observations": len(records),
        "n_usable": len(usable),
        "usable_cloud_max": USABLE_CLOUD_MAX,
        "first": records[0]["time"] if records else None,
        "last": records[-1]["time"] if records else None,
        "usable_dates": [(r["time"], r["cloud_cover"]) for r in usable],
    }
    if failure_date:
        before = [r for r in records if r["time"] < failure_date]
        usable_before = [r for r in usable if r["time"] < failure_date]
        out["last_observation_before_failure"] = before[-1]["time"] if before else None
        out["last_usable_before_failure"] = (
            usable_before[-1]["time"] if usable_before else None
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs" / "observation_gap.json")
    args = ap.parse_args()

    langtang = query_point(*LANGTANG)
    lang_summary = summarise(langtang, LANGTANG_FAILURE_DATE)

    # Robustness: the two published source positions differ by 1.1 km. Run the same
    # query at the alternative position and record whether the headline result moves.
    alt_summary = summarise(query_point(*LANGTANG_ALT), LANGTANG_FAILURE_DATE)
    alt_same = all(
        lang_summary[k] == alt_summary[k]
        for k in ("n_observations", "n_usable", "last_usable_before_failure")
    )

    print(f"Langtang source zone {LANGTANG[1]:.4f} N, {LANGTANG[0]:.4f} E")
    print(f"  EMIT L2A observations covering the point: {lang_summary['n_observations']}")
    print(
        f"  usable (< {USABLE_CLOUD_MAX:.0f}% cloud): {lang_summary['n_usable']}"
        f"  spanning {lang_summary['first']} to {lang_summary['last']}"
    )
    for date, cc in lang_summary["usable_dates"]:
        print(f"    {date}   cloud {cc:.0f}%")
    print(f"  last usable look before the 2026-08-26 failure: "
          f"{lang_summary['last_usable_before_failure']}")
    print(f"  alternative published position {LANGTANG_ALT[1]:.4f} N, "
          f"{LANGTANG_ALT[0]:.4f} E returns the same result: {alt_same}")

    sirmilik = {
        "latitude": SIRMILIK[1],
        "emit_latitude_limit": EMIT_LATITUDE_LIMIT,
        "emit_can_observe": abs(SIRMILIK[1]) <= EMIT_LATITUDE_LIMIT,
        "n_observations": 0,
        "reason": (
            "the ISS orbit reaches only about +/- 52 degrees, so EMIT never overflies "
            "this latitude; PRISMA acquires only within +/- 70 degrees, and EnMAP's "
            "proposal-driven tasked archive holds no systematic sea-ice coverage"
        ),
    }
    print(f"\nSirmilik {SIRMILIK[1]:.2f} N")
    print(f"  EMIT observations possible: {sirmilik['emit_can_observe']} -- {sirmilik['reason']}")

    payload = {
        "langtang": {"site": {"lon": LANGTANG[0], "lat": LANGTANG[1]},
                     "failure_date": LANGTANG_FAILURE_DATE,
                     "summary": lang_summary, "records": langtang,
                     "alt_position": {"lon": LANGTANG_ALT[0], "lat": LANGTANG_ALT[1],
                                      "summary": alt_summary,
                                      "same_headline_result": alt_same}},
        "sirmilik": sirmilik,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
