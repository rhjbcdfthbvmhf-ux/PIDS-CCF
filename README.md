# PIDS–CCF

Code for **Compound sensor-inspired augmentation with center–context fusion for robust hyperspectral image classification**.

Yuhan Liu, Guoxing Li, and Qianru Zhao

## Installation

```bash
git clone https://github.com/rhjbcdfthbvmhf-ux/PIDS-CCF.git
cd PIDS-CCF
python -m pip install -r requirements.txt
```

The training queues use Linux and PyTorch. Install the appropriate
[PyTorch CUDA build](https://pytorch.org/get-started/locally/) for your GPU.

## Data

Download Indian Pines, Pavia University, Salinas, Botswana and KSC from the
[benchmark data page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).
Place the first three image/label pairs in `data/raw/`, and Botswana and KSC in
`data/additional/`. File names and keys are specified in `src/data.py` and
[`revision/README.md`](revision/README.md).

## Run

```bash
# Main experiments
bash run_final_core_5seeds.sh
bash run_final_baselines_5seeds.sh
bash run_final_ablations_5seeds.sh
bash run_buffered_sensitivity_5seeds.sh

# Spatial validation and additional datasets
python revision/run_cv_queue.py
python revision/run_additional_queue.py

# Additional comparison models
python revision/fetch_author_models.py
python revision/run_native_queue.py --models ssftt dbda
python revision/run_native_gpu_queue.py

# Published spatial partitions
python revision/check_published_inputs.py --data-root data/raw
python revision/run_published_queue.py --data-root data/raw
```

Run the queues sequentially. The full evaluation commands and configurations are
in [`REPRODUCTION.md`](REPRODUCTION.md) and [`revision/README.md`](revision/README.md).

## Verify recorded results

```bash
python verify_release.py
```

This command checks the files and regenerates Tables 10 and 11 and their CSV
results using the Python standard library. Training data and a GPU are not
required for this check.

## License

[MIT](LICENSE). External data and model sources retain their original terms;
see [third-party notices](THIRD_PARTY_NOTICES.md).
