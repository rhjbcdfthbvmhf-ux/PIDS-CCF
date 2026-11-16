# PIDS–CCF reproducibility code

Implementation for *Compound sensor-inspired augmentation with center–context fusion for robust hyperspectral image classification*.

**Authors:** Yuhan Liu, Guoxing Li, and Qianru Zhao. **Release:** 1.0.0.

## Download and quick verification

Clone this repository or use **Code → Download ZIP**. A standalone
`PIDS-CCF-v1.0.0.zip` is also provided with release v1.0.0. Both contain source
code, launchers, configurations, spatial coordinates and recorded summaries.
From the extracted project directory run:

```bash
python verify_release.py
```

This standard-library check verifies file integrity and reconstructs Tables 10
and 11 and their full-condition CSV files from the recorded results. It requires
neither a GPU nor the original image data. Training and prediction-level
reproduction use the commands below and in `revision/README.md`.

## Install for training

Use Linux for the complete recorded training queues, with Python 3.11 and a
PyTorch/CUDA combination supported by the host. Create an isolated environment,
install an appropriate PyTorch build, then install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements record the experiment's library versions. GPU retraining is
separate from the lightweight verification above. Install instructions for
PyTorch are available at https://pytorch.org/get-started/locally/ .

## Reproduction map

| Experiment | Entry point | Recorded evidence |
|---|---|---|
| Main five-seed CCF and PIDS training | `run_final_core_5seeds.sh` | `results/core/` |
| Backbone comparisons | `run_final_baselines_5seeds.sh` | `results/baselines/` |
| Ablations and sensitivities | `run_final_ablations_5seeds.sh`, `run_buffered_sensitivity_5seeds.sh` | `results/sensitivities/` |
| Five-fold spatial blocks, Table 8 | `revision/run_cv_queue.py` | `revision/CV_AUDIT_AND_RESULTS.json` |
| Botswana and KSC, Table 9 | `revision/run_additional_queue.py` | `revision/ADDITIONAL_AUDIT_AND_RESULTS.json` |
| SSFTT, DBDA and S2Mamba, Table 10 | `revision/run_native_queue.py`, `revision/run_native_gpu_queue.py` | `revision/NATIVE_AUDIT_AND_RESULTS.json` |
| Released spatial partitions, Table 11 | `revision/run_published_queue.py` | `revision/PUBLISHED_AUDIT_AND_RESULTS.json` |

Follow the detailed execution order in `revision/README.md`, including the model
source downloader before the comparison queues. Run one training queue at a time.

## Data
Obtain Indian Pines, Pavia University and Salinas from the original public benchmark providers linked by https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes . Follow their usage and citation terms. Place the six MAT files in `data/raw/`; the required names and MATLAB keys are in `src/data.py`. No original dataset is redistributed here.

## Environment
The recorded numerical environment used Python, PyTorch 2.13.0+cu130, NumPy 1.26.4, SciPy 1.16.2 and scikit-learn 1.7.2. Install a PyTorch build compatible with the host GPU driver; CUDA builds are obtained from the official PyTorch distribution. CPU and different GPU/library versions can produce numerical differences.

## Core experiments
From this directory, set `PY` to the desired Python executable and run `bash run_final_core_5seeds.sh`. The script records five independent seed-specific splits, preprocessing parameters, training configurations, validation selection, checkpoints and predictions. It runs clean CCF, PIDS stage 1, PIDS stage 2, stage selection and the secondary router. Seeds are 23, 37, 51, 71 and 101. These are repeated seeded experiments, not k-fold cross-validation.

The backbone, ablation, and spatial-separation experiments are launched by `run_final_baselines_5seeds.sh`, `run_final_ablations_5seeds.sh`, and `run_buffered_sensitivity_5seeds.sh`. Run sequentially on an available GPU. Full patch separation uses Chebyshev buffer radius 10 for 11 × 11 patches.

## Inspection and recomputation
`src/audit_prediction_records.py` recomputes metrics from saved predictions and checks label/coordinate alignment and spatial separation. The three `aggregate_*.py` programs calculate seed-level summaries. Use each program's `--help` for its input-directory arguments. Average degradation draws within each seed before computing uncertainty over independent runs. `results/` contains the recorded aggregate snapshots for comparison; it does not replace raw-prediction recomputation.

The revision/ directory contains the added spatial-block validation, additional-scene evaluation and author-model comparison drivers, with separate protocols and audit scripts. See revision/README.md for their execution order and recorded evidence. Trained checkpoints and individual prediction files are not included in this archive.

## Aggregate commands
After completing the original queues, run:

```
python src/aggregate_final.py --server-root . --output final_aggregates/core
python src/aggregate_baselines.py --server-root . --output final_aggregates/baselines
python src/aggregate_sensitivities.py --server-root . --output final_aggregates/sensitivities
python src/audit_prediction_records.py --root . --family core --output final_aggregates/core_prediction_audit.json
python src/audit_prediction_records.py --root . --family baselines --output final_aggregates/baselines_prediction_audit.json
python src/audit_prediction_records.py --root . --family sensitivities --output final_aggregates/sensitivities_prediction_audit.json
python src/profile_models.py --output final_aggregates/model_profile --device cuda
```

Hardware profiling is a separate inference benchmark and depends on the host. Do not run it alongside active training. Compare regenerated aggregate values with the included recorded results; floating-point execution differences may affect retraining outcomes.

## License and citation

The authors' original code is distributed under the MIT license. Dataset and
third-party source terms are described in `THIRD_PARTY_NOTICES.md`. Software
citation metadata are provided in `CITATION.cff`. The associated manuscript title
is given at the top; publication metadata should be added when available.
