# Revision experiments

These experiments are separate from the original two-stage study. Do not pool their results with the original protocol.

## Data and author models
Place PaviaU.mat and PaviaU_gt.mat in ../data/raw. Place Botswana.mat, Botswana_gt.mat, KSC.mat and KSC_gt.mat in ../data/additional. The original benchmark provider is https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes . The additional scenes used here were downloaded from https://zenodo.org/records/22828439 . Follow the provider's usage and citation terms; this archive redistributes no scene imagery.

Run `python revision/fetch_author_models.py` from the project root to obtain the fixed author sources. Source repositories, version identifiers and expected checksums are recorded in third_party_sources.json. Third-party license terms apply to their respective sources. The archive does not sublicense third-party code.

SSFTT retains its author network and classifier, with 30 whitened PCA components fitted on training centers only. Perturbations are applied in the original spectral domain before the fixed PCA projection. DBDA retains DBDA_network_MISH and its classifier, with only the required tensor-axis conversion. S2Mamba retains the author model, using its patch-11 Pavia configuration and the official pure-PyTorch selective_scan_ref execution backend. No model-head replacement is used in these added comparisons.

## Running
Use the recorded Python dependencies. Spatial-fold and additional-scene runs use two CPU threads. SSFTT and DBDA were run on CPU. S2Mamba uses CUDA FP32, with AMP and TF32 disabled, and generates perturbations on CPU to match the same validation and test inputs. The GPU reference scan uses the same published scan function. Run queues sequentially from the repository root:

```
python revision/run_cv_queue.py
python revision/run_additional_queue.py
python revision/run_native_queue.py --models ssftt dbda
python revision/run_native_gpu_queue.py
```

The CPU queue accepts a model list and defaults to all three when the option is omitted. The commands above reproduce the recorded CPU and GPU allocation. Each S2Mamba run starts from its original seed for all 40 epochs. CPU and CUDA floating-point operations and random dropout can produce different training trajectories.

Each run is immutable: existing run directories are not overwritten, and queue logs use exclusive creation. Review a failed run before deciding how to resume it. The five-fold experiment uses 24-pixel spatial groups, 11-by-11 inputs, a ten-pixel center exclusion between roles, and fixed classwise caps of 128/64/256. Fold seeds are 23–27. Each role includes all nine classes. These are explicitly defined spatial groups, not the released folds of another paper. The three author-model comparisons use the first three of those same folds, with clean/PIDS pairs. The additional-scene runs use seeds 23/37/51 and the original 13.5/1.5/85 center-pixel split. Every revision run uses 40 epochs and three degradation draws per condition.

## Auditing
After each queue completes, run:

```
python revision/audit_cv_results.py
python revision/audit_revision_runs.py additional
python revision/audit_revision_runs.py native
python revision/make_native_result_objects.py
```

The auditors independently rebuild prediction metrics, verify ground-truth alignment, check paired degradation parameters, and examine preprocessing and split support. The table builder creates the additional-model LaTeX table and full-condition CSV from the verified native summaries and the same three CCF folds. Exact reconstruction of stratified split coordinates requires the recorded scikit-learn version. Summary uncertainty uses sample SD over independent seeds or spatial folds, after averaging draws within a run. Spatial folds are not treated as independent replicates for a t-test.

Recorded summaries and run configurations are included for all ten spatial-fold runs, twelve additional-scene runs, and eighteen native-model runs. Their audits respectively recomputed 160, 192, and 288 prediction files. Re-running training is required to regenerate checkpoints and predictions; those larger files are not embedded in this code archive. The native-model table compares CCF on the same first three folds, not its five-fold mean. Audit JSON files retain all evaluated conditions and models.

## Published spatial partitions

The folded benchmark uses the four Indian Pines, five Pavia University and five Salinas partitions released by Nalepa et al., https://doi.org/10.1109/LGRS.2019.2895697 . The authors link their release at https://tinyurl.com/ieee-grsl . The partition files in published_split_bank give the region coordinates in zero-based row-column order. Obtain the three image/label pairs from the original benchmark provider and place them in data/raw. The loader checks the source-file identities.

Each development region is processed as a separate image. A 7-by-7 window uses zero padding at region boundaries. All development regions are masked in the test image. Training and validation centers share development regions and have distinct labels by location. Testing uses every released labeled test center and the full scene class vocabulary. A fold pairs clean and PIDS training with the same split and test perturbations. Summaries first average the three draws at each degraded condition, then give the mean and sample standard deviation across partitions within a scene.

From the package root on a Linux CUDA host, run:

```
python revision/check_published_inputs.py --data-root data/raw
python revision/run_published_queue.py --data-root data/raw
python revision/make_published_result_objects.py revision/PUBLISHED_AUDIT_AND_RESULTS.json revision/result_objects
```

The queue runs the prediction audit after completing the 28 fits. Each fit saves the split, scaling parameters, validation history, model and predictions in revision/published_runs. The last command creates the manuscript table and a fold-level CSV covering all six conditions.

The recorded published-partition summaries cover all 28 fits, with 448 prediction files independently audited and 14 matched clean/PIDS pairs. Their configurations, coordinates, scaling parameters and validation histories are included under recorded/published_runs. These released partitions have overlapping test populations across folds, so their sample standard deviations describe partition variability and are not used in an independent-fold significance test.
