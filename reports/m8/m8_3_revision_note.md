# M8.3 revision note (2026-10-18)

Run 1 of `scripts/predict_baselines.py` (provenance: `m8_3_run1_provenance.json`) fitted the
TF-IDF + logistic-regression baseline with C in (0.1, 1, 10) and max_iter 300. The selected
model (C = 10, the largest value in the grid) stopped at the iteration limit without
converging. This was a validation-side diagnostic; the script prints no test numbers, and no
baseline test number had been seen when the revision was decided.

Revision (pre-registered as the only one): C grid widened to (0.1, 1, 10, 100), max_iter
raised to 1000, iterations recorded per C. All baseline dumps were regenerated.

Outcome (`m8_3_provenance.json`): best C stayed 10 (C = 100 scored lower), the fit converged
in 304 iterations, validation macro-F1 moved from 0.2712 to 0.2711. The majority,
stratified-random and length-only dumps are byte-identical to run 1.
