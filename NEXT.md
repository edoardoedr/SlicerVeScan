# Next steps

## Extension
- [ ] Module acknowledgement (`parent.acknowledgementText`: institutions, funding, VeScan citation) and contributors.
- [ ] Choose the extension category (`EXTENSION_CATEGORY` in the top-level `CMakeLists.txt`, now missing; needed for the Extensions Index / Extension Manager). The module itself has no category (top level of the module list).
- [ ] Optional: rename the local folder `SlicerVEScan` → `SlicerVeScan` (the GitHub repo is renamed; then update Slicer's additional module path).
- [ ] Test with a small sample case in `VeScanTest` (now: geometry and config only).

## VeScan-field
- [ ] At each new release: `python3 tools/sync_vescan.py <tag>`, review the reported config differences (field and thalweg), commit.
- [ ] Thalweg method (vescan-field 1.1.0): full run in Slicer 5.12 on a sample case, both bifurcations (zone / point), then Apply cut.

## Later
- [ ] VMTK method (`vmtkRadio` is disabled): postponed.
