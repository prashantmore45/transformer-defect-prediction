# M4 Stage 3 (Linux/WSL) -- LINKER Remeasurement Summary

- Original (GCC 6.3.0, Windows): 5/402 = 1.2438%
- Linux (GCC 15.2.0, WSL): 5/400 = 1.2500%

## Windows-PE relocation-overflow files, re-tested under Linux
All 4 files that failed with IMAGE_REL_AMD64_REL32 relocation errors under Windows MinGW-w64 (both GCC 6.3.0 and GCC 16.2.0) compile CLEAN under Linux GCC. This confirms the failure was a Windows PE/COFF target artifact, not a code defect.

## Transitions
- Still LINKER: ['s290051601', 's771232448', 's391527030', 's679217056']
- Was LINKER, resolved under Linux:
  (none)
- Not LINKER originally, now LINKER under Linux:
submission_id original_classification
   s895117405                   OTHER

## Agreement-filter files under Linux
linux_classification
COMPILES_CLEAN         95
OTHER_COMPILE_ERROR     3

## Extrapolation to full 30,000-file reserve (estimate)
~375 LINKER files expected under Linux GCC.