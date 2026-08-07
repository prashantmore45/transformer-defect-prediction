# Data Directory

**No data files are tracked by Git.** Everything here except this README and
the `.gitkeep` placeholders is excluded via `.gitignore`.

## Layout

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | Immutable inputs. Never edited by any script. | `notebooks/01_metadata_acquisition.ipynb` |
| `interim/` | Intermediate artefacts of the labelling pipeline. | Milestone 2+ |
| `processed/` | Final train / validation / test splits. | Milestone 2+ |

## Expected files in `raw/`

| File | Approx. size | Description |
|---|---|---|
| `codenet_cpp_metadata.parquet` | ~100 MB | All C++ submission records from CodeNet (metadata only, no source code) |
| `problem_list.csv` | ~200 KB | Problem-level metadata: name, source judge, time and memory limits |

## How to obtain the data

**Option A — shared Drive (recommended).**
Download both files from the group's shared folder into `data/raw/`.
Link: *(https://drive.google.com/drive/folders/1vmrRCWze_2fXzbgdyJsmxlxVdrjD0ZKk?usp=sharing)*

**Option B — regenerate from source.**
1. Download `Project_CodeNet.tar.gz` (7.8 GB) from IBM Data Asset eXchange
   to a location **outside this repository** (e.g. `D:\Datasets\CodeNet\`).
2. Extract only the `metadata/` subtree.
3. Run `notebooks/01_metadata_acquisition.ipynb`.

Full commands are in the notebook. Allow ~11 GB of free disk space.

## Notes

- `raw/` is immutable by convention. If a script corrupts a downstream file,
  re-derive it from `raw/` rather than re-downloading the archive.
- The source archive is kept outside the repository and is needed again in
  Milestone 2 to extract the selected source files. Do not delete it.
- Dataset: IBM Project CodeNet (Puri et al., 2021), licensed CDLA-Permissive-2.0.
  Full details in `docs/DATASET.md`.
