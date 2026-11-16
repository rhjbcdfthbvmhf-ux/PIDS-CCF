# Data and third-party sources

The MIT license covers the authors' original implementation and documentation.
External datasets, third-party model implementations and externally supplied
spatial partitions retain their providers' terms.

## Datasets and partitions

- Indian Pines, Pavia University, Salinas, Botswana and Kennedy Space Center:
  obtain the images and labels using the sources in the two README files. Original
  scene imagery and label MAT files are not redistributed in this package.
- Published spatial partitions: Nalepa et al.,
  [doi:10.1109/LGRS.2019.2895697](https://doi.org/10.1109/LGRS.2019.2895697),
  [author release](https://tinyurl.com/ieee-grsl). Coordinate records in
  `revision/published_split_bank/` document these external partitions. They are
  attributed to the benchmark authors and are not relicensed by this project.

## Comparison implementations

`revision/fetch_author_models.py` downloads the following separately. The
repository revisions, source paths and expected checksums are specified in
`revision/third_party_sources.json`. Downloaded sources are not embedded in the
release archive; consult each upstream repository for its own license and terms.

| Component | Upstream repository |
|---|---|
| SSFTT | https://github.com/zgr6010/HSI_SSFTT |
| DBDA | https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network |
| S2Mamba | https://github.com/PURE-melo/S2Mamba |
| Mamba reference scan | https://github.com/state-spaces/mamba |
| timm drop-path implementation | https://github.com/huggingface/pytorch-image-models |

The training adapters use these implementations with the input conventions
described in `revision/README.md`. This project's license does not grant rights
to third-party sources or datasets.
