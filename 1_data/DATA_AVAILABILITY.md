# Data Availability

Everything this submission uses is public and requires no credentials. Catalogue
counts are stated as of **2026-08-29**; live catalogues grow.

## Tanager (primary data)

| | |
|---|---|
| Scene | `20250606_181248_58_4001` — Sirmilik National Park area, Nunavut, 2025-06-06 18:12:48 UTC |
| Collection | `snow-ice`, Tanager Open STAC catalogue: https://www.planet.com/data/stac |
| Asset used | `ortho_sr_hdf5` (HDF-EOS5, ~883 MB): `surface_reflectance` + `surface_reflectance_uncertainty`, 426 bands 376–2499 nm, 30 m ortho grid, EPSG:32617 |
| Access | public Google Cloud Storage, no authentication — `python -m tanager_cryo.fetch --scene 20250606_181248_58_4001` |
| Licence | © 2025 Planet Labs PBC, CC-BY 4.0 |

The catalogue held 153 unique scenes across 9 collections as of 2026-08-29.
`python -m tanager_cryo.glacier_check` audits every scene footprint against the
OpenStreetMap `natural=glacier` layer (result: zero intersections; output in
`5_outputs/glacier_audit.json`).

## Sentinel-2 (independent validation)

| | |
|---|---|
| Item | `S2C_17XMB_20250606_0_L2A`, 2025-06-06 17:43:35 UTC — 29 minutes before the Tanager pass |
| Source | Earth Search STAC (AWS `sentinel-2-l2a` COGs, element84), windowed HTTP range reads only |
| Item JSON | committed at `1_data/raw/s2_item.json` |
| Licence | Copernicus Sentinel data, ESA — free and open |

## EMIT (observation-gap audit)

Granule metadata (footprints, cloud cover, dates) queried live from NASA CMR
(`https://cmr.earthdata.nasa.gov/search/granules.json`, collection `EMITL2ARFL`),
which needs no authentication. No EMIT granule is downloaded. Output in
`5_outputs/observation_gap.json`, including the robustness check at the alternative
published source position.

ICESat-2 (ATL07/ATL10) was evaluated as a validation source and abandoned: all
three tracks crossing the scene in June 2025 are flagged 100% cloud-covered.

## Optical constants and instrument response (committed in `optical_constants/`)

- Ice: Warren & Brandt (2008), *J. Geophys. Res.* 113, D14220 — `ice_warren_brandt_2008.dat`
- Liquid water: Segelstein (1981), MSc thesis, UMKC — `water_segelstein_1981.txt`, `water_abs_segelstein.npy`
- Sentinel-2A spectral response functions: ESA COPE-GSEG-EOPG-TN-15-0007 — `S2A_MSI_spectral_responses.xlsx`, `s2a_srf.npz`
- Tanager per-band medians derived from the scene itself: `tanager_wavelengths.npy`, `tanager_uncertainty_median.npy`, `tanager_snr_median.npy`, `tanager_good_wavelengths.npy`

## OpenStreetMap

Glacier polygons queried through the Overpass API (`natural=glacier`), © OSM
contributors, ODbL. OSM is not a complete glacier inventory; the audit script says
so, and the analysed scene was additionally checked by inspection.

## Ground truth

There is no in-situ campaign at this scene and date — no buoys, no coincident
aerial survey. The retrieval is therefore trained on a physical forward model and
validated against same-day Sentinel-2 (see the memo, §4), with the absence of field
truth stated rather than papered over.
