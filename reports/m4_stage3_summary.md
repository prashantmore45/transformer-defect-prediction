# M4 Stage 3 -- LINKER Remeasurement Summary

## Two exclusions from the nominal 500 (Stage 1)
- s198272745: confirmed cross-split dedup drop; ORIGINALLY LABELED LINKER
- s363457044: untraceable exclusion; originally labeled OTHER

## Denominator note
Original 1.24% matches 5/402 = 1.2438% (reproducing-failure files only, excluding 98 agreement-filter discards) far more closely than 5/500. This denominator is inferred from the arithmetic match, not yet confirmed against the docs/LABELING.md §5 text directly.

## Rates
- Original (GCC 6.3.0): 5/402 = 1.2438%
- New (GCC 16.2.0), same denominator basis: 8/400 = 2.0000%
- New (GCC 16.2.0), full working sample -- NOT comparable, listed only to show the difference denominator choice makes: 8/498 = 1.6064%

## Transitions (reproducing-failure subset only)
- Still LINKER under both compilers: ['s290051601', 's771232448', 's391527030', 's679217056']
- LINKER under GCC 6.3.0, resolved to something else under GCC 16.2.0:
  (none)
- Not LINKER under GCC 6.3.0, now LINKER under GCC 16.2.0:
submission_id original_classification
   s966843997                   OTHER
   s826249906                   OTHER
   s103319025                   OTHER
   s634326722                   OTHER

## Agreement-filter files (originally compiled clean) under GCC 16.2.0
new_classification
COMPILES_CLEAN         91
OTHER_COMPILE_ERROR     7

## Extrapolation to the full 30,000-file COMPILE_ERROR reserve
**ESTIMATE, not a measurement** -- assumes this pilot's rate generalizes uniformly, which has not been checked, and assumes the reproducing-failure proportion (~80%) also holds at full-reserve scale.
- Estimated LINKER count (new rate basis): ~600
- Estimated LINKER count (original rate basis, for reference): ~373

This is the number the section-1 contingency decision (train-anyway-with-caveat vs. merge LINKER into SEMANTIC) depends on. This script does not make that call.