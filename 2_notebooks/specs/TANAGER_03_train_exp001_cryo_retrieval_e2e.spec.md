# Spec: TANAGER_03_train_exp001_cryo_retrieval_e2e.ipynb

## Goal
Let a competition judge reproduce the entire submission end to end on Colab with zero
credentials: train the cryosphere retrieval, evaluate accuracy and calibration, apply it
to the real Tanager scene, validate independently against same-day Sentinel-2, and
re-check the memo's headline numbers.

## Platform and resources
- Platform: Colab (also runs locally)
- Runtime: CPU
- GPU: N/A — CPU is sufficient
- Rationale: the model has only 213,644 parameters and the training set is 150k × 221
  spectra (265 MB). A single training step takes 0.77 ms; 80 epochs of pure compute is
  ~20 s. The bottleneck is entirely network I/O (the 883 MB Tanager scene and the
  CMR/STAC queries), not compute. Uploading data to a rented GPU would take longer than
  the training itself.
- Estimated time: ~3 min with `DRY_RUN = True`; ~25 min full (of which ~15 min is
  downloading)

## Cell outline
| Cell # | Purpose | Main actions |
|--------|---------|--------------|
| 1 | Setup & Imports | imports, `_cell_times` init, `cell_start/cell_end` |
| 2 | Config | `DRY_RUN`, repo acquisition (clone or local detection), path constants, seed, runtime detection |
| 3 | Install & Smoke Test | pip install pinned deps, import checks (incl. `xarray-hyperspectral` engine registration) |
| 4 | The observation gap | CMR query for EMIT coverage of the Langtang source zone; print usable observations and gap in months |
| 5 | Glacier audit | Overpass spot check (DRY_RUN checks the 7 Snow/Ice scenes), confirming the open catalogue holds no glacier |
| 6 | Fetch Tanager scene | `tanager_cryo.fetch` downloads the Sirmilik ortho SR (DRY_RUN fetches the thumbnail only) |
| 7 | Band selection from uncertainty | derive the 221 bands from the scene's own SNR spectrum; plot SNR curve and threshold |
| 8 | Forward model sanity | five physical checks (grain size ↑ → NIR ↓, pond depth ↑ → red ↓, impurities darken the visible only, …) |
| 9 | Synthetic training set | generate synthetic spectra (including the smooth model-error perturbation); print coverage statistics |
| 10 | Train | train the heteroscedastic MLP, tqdm shows per-epoch NLL |
| 11 | Evaluate | accuracy (incl. conditional) + calibration (var(z), 68/95% coverage) |
| 12 | Retrieve on the real scene | apply to Sirmilik; print forward-model error vs instrument σ and explained fraction |
| 13 | Independent validation | same-day Sentinel-2 10 m classification / linear dark fraction, aggregated 3×3 |
| 14 | Sensor comparison | degradation experiment, Tanager 221 bands vs Sentinel-2's 10 (reduced size in DRY_RUN) |
| 15 | Figures | write all figures to `5_outputs/figures/` |
| 16 | Verify | run `tanager_cryo.verify` against the memo's headline numbers |
| 17 | Summary | very short handover |

## Key hyperparameters
- learning_rate: 2e-3 (AdamW, weight_decay 1e-4, CosineAnnealingLR)
- batch_size: 512 (same on CPU/GPU; the model is tiny, not VRAM-bound)
- num_epochs: 80 (DRY_RUN: 8)
- n_train: 150000 (DRY_RUN: 8000)
- optimizer / scheduler / loss / metric: AdamW / cosine / Gaussian NLL / RMSE + var(z) calibration
- random seed: 0
- SNR threshold: 30
- MODEL_ERROR_RMS: 0.03

## Data paths
- Inputs:
  - `1_data/raw/sirmilik/20250606_181248_58_4001_sr.h5` (public GCS, no credentials)
  - `1_data/raw/s2_item.json` (Earth Search STAC item; assets are public AWS COGs)
  - `1_data/optical_constants/*` (committed with the repo)
- Outputs:
  - `4_models/cryo_retrieval.pt`, `cryo_retrieval_meta.json`
  - `5_outputs/sirmilik_retrieval.nc`, `s2_validation.nc`, `s2_validation.json`
  - `5_outputs/sensor_comparison.json`, `observation_gap.json`, `evaluation.json`
  - `5_outputs/figures/fig1..fig6*.png`

## Data specification
- Input shape / dtype: Tanager ortho SR HDF-EOS5, `surface_reflectance` (426, 804, 869)
  float32; `surface_reflectance_uncertainty` with the same shape
- Total volume: single scene 883 MB; Sentinel-2 is read as windowed COGs, never a full tile
- Batch shape into the model: (B, 442) — 221-band reflectance + 221-band log uncertainty
- Label / target shape: (B, 6) float32 — f_solid, f_pond, f_water, log10 L, pond_depth_m, lap_load
- I/O: HDF5 read locally; Sentinel-2 via HTTP range requests (rasterio windowed reads)
- Note: 58 bands are flagged by `good_wavelengths` as water-vapour absorption; the southern
  half of the scene is land, which the three-endmember model cannot describe and the χ²ᵥ
  test rejects automatically

## Data staging and preprocessing strategy
- Separate preprocess notebook needed: no — training data are synthetic spectra generated
  in memory, not a disk dataset
- Many small raw files or Google Drive: no — a single 883 MB HDF5 on public GCS
- Cache / archive form: N/A (synthetic data generated in memory, ~265 MB)
- Drive artifact: N/A — Google Drive is not used
- Colab local staging: the scene downloads to `/content/PBC_Tanager_Open_Data/1_data/raw/`
- CPU-heavy steps that must never run inside the training loop: optical-constant
  interpolation and endmember spectrum generation happen once, outside the loop

## Packages and wheels
- requirements: see `requirements.txt` (numpy>=2.0, xarray>=2024.1,
  xarray-hyperspectral>=0.1, torch>=2.2, rioxarray>=0.15, rasterio, geopandas>=1.0,
  matplotlib>=3.8, openpyxl>=3.1, scipy, h5py, netCDF4)
- wheelhouse: N/A — everything has PyPI wheels
- packages needing pre-built wheels: none
- install smoke test: `import torch, xarray, rasterio, h5py`; confirm the
  `xr.open_dataset(..., engine="tanager")` engine is registered

## Verification contract (implementations must not remove or weaken these)
- Band selection must be **derived from the scene's own SNR spectrum**, never hard-coded
  wavelength windows
- Training noise must include the **smooth** (low-order Legendre) model-error
  perturbation; it must not be replaced with white noise
- Evaluation must report **both unconditional and conditional** accuracy (conditional
  threshold: host endmember > 0.25)
- Calibration must report var(z) and empirical 68/95% coverage, not RMSE alone
- Retrieval must run the **forward-reconstruction residual test**, with an error budget
  combining instrument σ and model error in quadrature; instrument σ alone is not allowed
- Sentinel-2 validation must produce **both the hard classification and the
  threshold-free linear** reference, and state the hard classifier's low bias on water
- The cross-sensor comparison must hold scene / forward model / architecture / schedule /
  seed fixed, varying only the band set
- `5_outputs/s2_validation.json` and `sensor_comparison.json` must be produced
- `tanager_cryo.verify` must run at the end and display its result; it may not be skipped
- The model-error perturbation must never be removed to make the numbers look better

## Prerequisites
- Prerequisite notebooks: none
- Required packages: see above

## notebook-rules sections that apply
- Naming conventions
- First-screen execution contract
- Cell structure and timing
- Summary cell format
- Dry Run mode
- Cell independence
- tqdm progress display
- Package and wheel preparation
- (GPU auto-detection: this notebook is CPU-only; Cell 2 still detects and prints the
  runtime but does not adjust batch size)

## Pipeline efficiency requirements
N/A — CPU notebook. 213,644 parameters, 80 epochs ≈ 20 s of compute; the bottleneck is
network I/O.

## Design constraints (things not to do)
- Do not add W&B / MLflow logging
- Do not change the Summary cell format
- Do not add augmentations / metrics / endmembers the spec does not mention
- Do not hard-code absolute paths (compose Colab paths from a `REPO_ROOT` variable)
- Do not require any credentials — the whole flow must run with zero logins (ICESat-2
  needs an Earthdata token; it was excluded and must not come back)
- Do not re-implement logic that already exists in `3_src/tanager_cryo/` inside the
  notebook — always import
- Do not drop any item of the verification contract to shorten the runtime
