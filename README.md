<p align="center"><img src="docs/logo/vescan_logo_512.png" alt="VeScan logo" width="160"></p>

# SlicerVeScan

**VeScan** — a toolkit for vessel tree cutting, anatomical branch assignment, and crossing-vessel characterization.

Given a set of vessel segmentations (arteries, veins, pulmonary airways) as input, VeScan turns the raw vessel tree into a fully labeled anatomical graph. Starting from the segmentation masks and the network's own continuous output (a ranked store written by [haversack](https://github.com/edoardoedr/haversack)), it:

- builds a **hierarchical centerline tree** (trunk → bifurcations → branches) from the segmentation network's continuous field, **without VMTK**;
- assigns every branch/bifurcation to its corresponding **anatomical segment** (lobar bronchus/artery/vein, main trunk, ...);
- detects **crossing vessels** (branches that cross into a neighbouring pulmonary lobe), together with the **interlobar fissures** they cross;
- **cuts the vessel tree** to a chosen number of generations.

**SlicerVeScan** brings VeScan into [3D Slicer](https://www.slicer.org). The computation is the core library [VeScan-field](https://github.com/edoardoedr/VeScan-field) (Python package `vescan-field`): the extension loads the inputs with Slicer, runs the VeScan stages on them and shows every result in the 3D view as soon as it is written. Two methods compute the centerlines, both in the same package: **Field** (`vescan_field`, VeScan-field's own tracer) and **Thalweg** (`vescan_thalweg`, the centerlines of Mike Halle's [thalweg](https://github.com/mhalle/thalweg)); every other stage is shared. Outputs, file names and colour tables are the same as VeScan-field's, whatever the method.

![Arteries, veins and airways labelled by anatomical segment in 3D Slicer](docs/screenshots/overview.png)

## Table of contents

- [Inputs: the haversack outputs](#inputs-the-haversack-outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Develop](#develop)
- [Credits](#credits)

## Inputs: the haversack outputs

VeScan needs the per-patient folder written by `batch_segment.py` of [edoardoedr/haversack](https://github.com/edoardoedr/haversack), a fork of Mike Halle's [haversack](https://github.com/mhalle/haversack). The segmentation itself is not done in Slicer: run haversack first (it needs its own environment with PyTorch and a GPU), then open its folder here.

| Module input | haversack file | Required |
|---|---|---|
| Artery, Veins, Airways | `lung_arteries.nii.gz`, `lung_veins.nii.gz`, `lung_airways.nii.gz` | the structures you run |
| Airway wall | `lung_airways_wall.nii.gz` | no (added to the airways) |
| Field store | `lung_vessels.duckn.zip`, the **ranked store**: for every voxel, the classes in rank order and how many logits each one is behind the winner | yes |
| Lobes | the five lobe masks `lung_{upper,middle,lower}_lobe_{right,left}.nii.gz` (or one label map) | for Preprocess and Anatomy |
| CT | the CT volume | no: only the reference grid, otherwise the grid of the first structure mask |

Each input is either a file or a node already in the scene (for the structures, a segment of a segmentation).

## Installation

Requires 3D Slicer 5.12 or newer (Python 3.12, numpy 2.4, VTK 9.6): `vescan-field` 1.1 needs Python ≥ 3.12, so 3D Slicer 5.10 is no longer supported. SlicerVeScan is not in the Slicer Extensions Index yet. To use it from the source:

1. Clone this repository.
2. In Slicer: *Edit → Application Settings → Modules → Additional module paths*, add the `VeScan/` folder of the clone and restart Slicer.
3. Open the module **VeScan** (top level of the module list, or search it with the module finder).

The core, `vescan-field`, is installed on first use: at the first Run (or the first lobe loading) the module asks to install it into Slicer's Python, at the version pinned in [`VeScan/Resources/requirements.txt`](VeScan/Resources/requirements.txt), keeping Slicer's numpy. When a new version of the extension requires a newer `vescan-field`, the module asks to update it (then restart Slicer). No git is needed: `vescan-field` and its dependencies that are not on PyPI ([thalweg](https://github.com/edoardoedr/thalweg), [rankfield](https://github.com/mhalle/rankfield), [labelfield](https://github.com/mhalle/labelfield)) are installed from their GitHub archives, listed in `requirements.txt`, without pip resolving them (`--no-deps`); their PyPI dependencies, in [`requirements-deps.txt`](VeScan/Resources/requirements-deps.txt), are installed after. No VMTK and no GPU are needed.

## Usage

![Inputs of one patient, loaded and shown in 3D](docs/screenshots/inputs.png)

1. **Inputs.** Set each input as *File* or *Scene*. A file is loaded and shown in 3D as soon as its path is set; a scene segment is shown when selected (only the chosen segments of a segmentation).
   - **Lobes**: select the five lobe files together (or one label map) and check the table: each lobe is mapped to a segment automatically by its name (`RUL` or `lung_upper_lobe_right`). A lobe can be left to *None*: VeScan then runs on the other lobes only.
2. **Method.** Choose the method: *Field* or *Thalweg*; with Thalweg also how a bifurcation is represented, *Zone (as VMTK)* (generations comparable with VMTK's) or *Point (as Field)* (`pipeline.centerlines.bifurcations`). Then the structures (artery, veins, airways) and the stages: *Preprocess surfaces*, *Centerline extraction*, *Anatomy*. Surface labels are always computed.
3. **Output.** Choose the output folder, or leave it empty to work in a temporary folder that is removed when the scene is closed (save the scene to keep the results). The folder is required when *Preprocess* or *Centerlines* is off: the results of a previous run are read from it.
4. **Run.** The progress bar follows the stages, the log shows VeScan's messages, and each output appears in 3D when it is written: lobes and fissures, surfaces, centerline trees, anatomical labels. *Cancel* stops after the current step.
5. **Cut tree.** One slider per structure sets the number of generations to keep. *Cut in lobes* limits the cut to the checked lobes (all by default): the branches that reach another lobe are kept whole (needs *Anatomy*). *Live preview* hides the deeper generations of the centerlines while you move the sliders; *Apply cut* runs VeScan's clip stage and shows the clipped surfaces (optionally capped) with the cut points.

![Cutting the trees: live preview, then Apply cut](docs/screenshots/clipping.gif)

6. **Results.**
   - **Show**: labelled surfaces, centerlines and cut points per structure, lobes / fissures, colour legends.
   - **Color by**: anatomical segment, lobe, crossing type or generation, with VeScan's colour tables; the legend is shown in the 3D view.
   - **Lobes color by**: the lobe mesh by artery, vein or bronchus lobe segment, or by fissure contact.
   - **Crossing vessels**: one row per crossing (structure, type, lobes, length, angle); click a row to centre the views on it. *Export statistics CSV* writes all the columns of VeScan's `statistics.csv`, every structure in one file.

The state of the module is saved with the scene: reopening a saved scene restores the inputs, the lobe table, the Cut tree and the Results. *Apply cut* works again as long as the run's files are still in the output folder.

## Results

Patient R01-015, coloured by `AnatomicalSegmentPoint` with the interlobar fissures: the multilobar vessels are the ones that cross a fissure.

| | Centerlines | Labelled surfaces |
|---|---|---|
| Arteries | ![Artery centerlines](docs/screenshots/centerlines_artery.png) | ![Artery surface](docs/screenshots/surfaces_artery.png) |
| Veins | ![Vein centerlines](docs/screenshots/centerlines_veins.png) | ![Vein surface](docs/screenshots/surfaces_veins.png) |
| Airways | ![Airway centerlines](docs/screenshots/centerlines_airways.png) | ![Airway surface](docs/screenshots/surfaces_airways.png) |
| All | ![All centerlines](docs/screenshots/centerlines_all.png) | ![All surfaces](docs/screenshots/surfaces_all.png) |

| Lobes by artery segment | Fissure contacts |
|---|---|
| ![Lobes coloured by artery lobe segment](docs/screenshots/lobes_3d.png) | ![Fissure contacts](docs/screenshots/fissures_3d.png) |

| Crossing vessels | Tree cut to 6 generations |
|---|---|
| ![Crossing vessels coloured by type, with the crossings table](docs/screenshots/crossings.png) | ![Clipped surfaces with the cut points](docs/screenshots/cut_applied.png) |

## Outputs

The output folder has VeScan-field's layout, written with its own functions:

```
<output folder>/
├── vescan_config.json            # the config of this run
├── pipeline.log
├── surfaces/                     # lobe surfaces, lobe_segments.vtk, lobe_fissures.vtk, structure surfaces
└── final_export_<structure>/     # anatomical_segments.vtk, branch_labeled_surface.vtk, statistics.csv/json, ...
                                  # clipped_surface.vtk and the cut points after Apply cut
```

Every file and scalar array is described in the [VeScan-field README](https://github.com/edoardoedr/VeScan-field#visualizing-results-in-3d-slicer) and in its [docs](https://github.com/edoardoedr/VeScan-field/tree/main/docs), one page per stage.

## Configuration

Every VeScan parameter that is not in the UI comes from the method's bundled config, a copy of VeScan-field's example with the standard values, loaded and validated by the method's package:

| Method | Config | Copy of | Validated with |
|---|---|---|---|
| Field | [`vescan_field.json`](VeScan/Resources/vescan_field.json) | `configs/field/example.json` | `vescan_field.config` |
| Thalweg | [`vescan_thalweg.json`](VeScan/Resources/vescan_thalweg.json) | `configs/thalweg/example.json` | `vescan_thalweg.config` |

The module sets only the output folders, the store file, the structures, the lobes, the stages, the bifurcations (Thalweg) and the cut. The config of each run is saved as `vescan_config.json` in the output folder.

## Develop

1. Slicer: *Edit → Application Settings → Developer → Enable developer mode*, restart.
2. *Application Settings → Modules → Additional module paths*: add the `VeScan/` folder of this repo, restart.
3. Optional, to work on the core too: editable install from a local clone of VeScan-field, with Slicer's numpy pinned (otherwise pip may replace Slicer's numpy and break Slicer):
   `/Applications/Slicer.app/Contents/bin/PythonSlicer -m pip install -e <VeScan-field clone> "numpy==2.4.6"`
4. Edit `VeScan/VeScan.py` or `VeScan/Resources/UI/VeScan.ui` (*Edit UI* opens Qt Designer), then *Reload* in the module panel.

**New VeScan-field release**: after its tag is pushed, run `python3 tools/sync_vescan.py <tag>` (e.g. `v0.1.2`). It checks the tag's version, writes `requirements.txt` and `requirements-deps.txt` from the release's dependency tree (users with an older `vescan-field` are then asked to update), copies the colour tables and lists the differences between the release's `configs/field/example.json` and `vescan_field.json`, and between `configs/thalweg/example.json` and `vescan_thalweg.json`, to review by hand.

Open tasks are in [NEXT.md](NEXT.md).

## Credits

Thanks to [Mike Halle](https://github.com/mhalle):

- [haversack](https://github.com/mhalle/haversack) is the segmentation pipeline whose outputs VeScan reads. VeScan uses it through the fork [edoardoedr/haversack](https://github.com/edoardoedr/haversack).
- The field-based approach to centerline extraction (centerlines computed from the network's continuous output instead of the binary mask) is his. He implements it in [thalweg](https://github.com/mhalle/thalweg), a field-based successor of VMTK. The Field method has its own implementation; the Thalweg method uses thalweg itself (through the fork [edoardoedr/thalweg](https://github.com/edoardoedr/thalweg)) for the centerlines, the graph and the VMTK-style branch split.
- The ranked store is read and decoded with [rankfield](https://github.com/mhalle/rankfield) (Apache-2.0).
