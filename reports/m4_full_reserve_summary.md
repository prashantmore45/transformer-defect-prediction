# M4 Full-Reserve Summary

Total files in COMPILE_ERROR reserve: 29876

## Full classification breakdown
classification
SEMANTIC               12571
AGREEMENT_FILTER        6090
SYNTAX                  5587
UNRECOGNIZED            4829
MISSING_DEPENDENCY       368
LINKER                   320
COMPILATION_TIMEOUT      111

## Tier-2 leaf counts (feed M6 dataset assembly)
classification
SEMANTIC    12571
SYNTAX       5587
LINKER        320
Total labelled: 18478 (61.8% of reserve)

## Discard breakdown
classification
AGREEMENT_FILTER       6090
MISSING_DEPENDENCY      368
COMPILATION_TIMEOUT     111
Agreement-filter rate: 6090/29876 = 20.4% (pilot: 19.6%)

## Needs review (not yet resolved)
classification
UNRECOGNIZED    4829
See reports\m4_full_reserve_unrecognized.csv and reports\m4_full_reserve_ambiguous.csv for the full lists.

## LINKER sanity check
Measured: 320 (pilot extrapolation: ~375)