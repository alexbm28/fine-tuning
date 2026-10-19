# Compute allocation introduction illustration

The figure combines a schematic two-stage procedure with the original experiment's two selected numerical cases. It is designed at the supplied chapter's **150 mm text width**, with Liberation Serif text and Computer Modern mathematics, matching the supplied plotting cell. No phase-retrieval introduction source was supplied.

## Files and use

- `outputs/compute_allocation.pdf` and `.svg`: vector artwork, with no embedded raster images.
- `outputs/compute_allocation.png`: 600 dpi preview, 3543 × 2314 pixels.
- `plot_compute_allocation.py`: plotting only; reads cached values without generating data or training.
- `caption.tex`: approximately 121-word LaTeX-ready caption.
- `figure.tex`: insertion snippet, with paths relative to the repository root; requires `graphicx` and `amsmath`.
- `validate_numerics.py`: independent numerical checks; reuses the archived datasets.
- `outputs/numerical_checks.json` and `outputs/plot_record.json`: detailed validation, numerical results, source hashes, and plotting settings.

From this folder, using Python 3.12 with the packages in `requirements.txt` and the Liberation Serif font installed:

```bash
python plot_compute_allocation.py
python validate_numerics.py
```

Both scripts also accept invocation by absolute path from another working directory. The plotting script supports `--data-dir` and `--output-dir`. The included cache is sufficient to replot; the training sweep does not need to run again.

## Source and provenance

The user's two uploaded cells are preserved byte-for-byte in `source/original_compute_cell.py` and `source/original_plot_cell.py`; hashes are in `source/source_hashes.json`. The chapter, especially Sections 4.2, 4.3 and 4.5.4, supplies the mathematical and experimental specification. Its original PDF was not modified.

Historical saved arrays were not supplied. The complete original computation cell was therefore run **unchanged once** in this new folder, including its exact RNG ordering, parameters, thin-SVD treatment, and stable log-domain evaluator. `neurips_figure1_lanczos_data/` contains that run's saved curves, metadata and summaries; `outputs/original_compute.log` records its output. `data/original_datasets.npz` archives the resulting deterministic datasets, including their first-80 subsets. These are results regenerated from the supplied original source, not a claim of bit-identical agreement with unavailable historical arrays. The middle case was retained in the original cache but omitted from the illustration.

To rebuild the cache if it has been lost, run `python source/original_compute_cell.py` from this folder in a fresh copy, then the two commands above. Running the original cell again overwrites its generated cache, so ordinary restyling should use only the plotting script.

## Numerical results and checks

Both selected panels use `d = 400`, ridge penalties and step sizes `0.05`, and the same budget `B = 2,097,152,000 = 65,536 × 80d`. Each curve contains all **16,385** feasible schedules. The horizontal coordinate is compute percentage, not iteration percentage. The values are the unnormalised squared update error in the fine-tuning Hessian geometry.

| `(n, n')` | Optimal `t` | Pretraining steps `k` | Fine-tuning steps `ell` | Pretraining compute | Minimum squared update error |
|---|---:|---:|---:|---:|---:|
| `(80, 320)` | 1,404 | 5,616 | 14,980 | 8.5693359375% | 5.1230755 × 10⁻³⁶ |
| `(320, 80)` | 12,820 | 12,820 | 14,256 | 78.2470703125% | 5.5659101 × 10⁻³³ |

All checks passed: exact integer budgets and global argmins across all 98,307 cached schedules; original parameters and nested datasets; moderate-step residuals versus direct GD; both full-budget endpoint identities; admissible step sizes; and independent extended-precision residuals at the selected minima and nearby allocations. Maximum relative direct-GD discrepancy is below `1e-15`; maximum independent log10 discrepancy is `7.11e-15`. Extended-precision checks use the original double-precision thin-SVD factors. No objective clipping, normalisation, smoothing, or arbitrary error floor is used. Every selected curve point is retained in the vector paths.

The PDF/SVG contain vectors only. The PDF, PNG and full LaTeX insertion were inspected at 150 mm width; labels do not overlap curves, both numerical panels share vertical limits, and LaTeX reports no missing glyphs or overfull boxes. The 121-word caption and schematic correctly distinguish the finite update from the fixed update between empirical optima. Original source files, the original figure, and thesis text were left unchanged.
