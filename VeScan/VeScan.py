import contextlib
import csv
import dataclasses
import importlib.metadata
import importlib.util
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import time
import types
from typing import Annotated

import numpy as np
import qt
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    Choice,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode


STRUCTURES = ["artery", "veins", "airways"]
SEGMENT_INPUTS = STRUCTURES + ["airwayWall"]  # scene source: a segment of a segmentation (qMRMLSegmentSelectorWidget)
INPUTS = ["ct", "artery", "veins", "airways", "airwayWall", "lobes"]
PATH_INPUTS = ["ct", "artery", "veins", "airways", "airwayWall"]  # one path each; lobes: several files
SEGMENTATION_FILES = "Segmentations (*.nii.gz *.nii *.nrrd *.seg.nrrd)"
LOBES = ["RUL", "RML", "RLL", "LUL", "LLL"]
# Color by combo -> (array, VeScan colour table; None = continuous Rainbow)
COLOR_BY = [
    ("AnatomicalSegmentPoint", "ColorTableVessels_EN"),
    ("LobeLabelPoint", "ColorTableLobeLabel_EN"),
    ("CrossingTypePoint", "ColorTableCrossings_EN"),
    ("Generation", None),
]
COLOR_TABLES_DIR = pathlib.Path(__file__).parent / "Resources" / "ColorTables"
LOBES_TABLE = "ColorTableLobesFissures_EN"
# Lobes color by combo -> (cell array of lobe_segments.vtk, its values in LOBES_TABLE): one labelling per vessel
# type, or the fissure contacts; the legend lists only these values
LOBES_COLOR_BY = [
    ("LobeSegment_Artery", [40, 41, 42, 43, 44]),
    ("LobeSegment_Vein", [30, 31, 32, 33, 34]),
    ("LobeSegment_Airway", [50, 51, 52, 53, 54]),
    ("FissureContact", [0, 60, 61, 62, 63]),
]
FISSURE_VALUES = [60, 61, 62, 63]  # FissureContact of lobe_fissures.vtk
PREVIEW_ARRAY = "CutPreviewGeneration"  # centerline cell array of the live preview (VeScanLogic.previewCut)
CROSSING_COLUMNS = ["ID", "structure", "type", "lobes", "length mm", "angle deg"]

# method -> its package (both in the vescan-field distribution, same API and output files; thalweg differs only in
# the store, the margin and the centerlines stage) and its config: every parameter is in this JSON, the UI
# overwrites only a few keys (VeScanLogic.buildConfig)
METHOD_PACKAGE = {"field": "vescan_field", "thalweg": "vescan_thalweg"}
RESOURCES_DIR = pathlib.Path(__file__).parent / "Resources"
CONFIG_PATHS = {method: RESOURCES_DIR / f"{package}.json" for method, package in METHOD_PACKAGE.items()}
BIFURCATIONS = ["vmtk", "point"]  # Thalweg's pipeline.centerlines.bifurcations, in the order of bifurcationsCombo
MIN_PYTHON = (3, 12)  # vescan-field >= 1.1 (requires-python), i.e. 3D Slicer >= 5.12
# vescan-field and its git dependencies as GitHub archives (--no-deps), then their PyPI dependencies; both written
# by tools/sync_vescan.py
REQUIREMENTS_PATH = RESOURCES_DIR / "requirements.txt"
REQUIREMENTS_DEPS_PATH = RESOURCES_DIR / "requirements-deps.txt"
AIRWAY_WALL_CLASS = "lung_airways_wall"
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])
# node attributes saved with the scene, to find the module's nodes again when a scene is loaded
INPUT_ATTRIBUTE = "VeScan.Input"        # file input name (value of _inputs) ...
INPUT_KEY_ATTRIBUTE = "VeScan.InputKey"  # ... and the path(s) it was loaded from (JSON)
MODEL_ATTRIBUTE = "VeScan.Model"        # "<kind>/<structure>" (key of _models)
TABLE_ATTRIBUTE = "VeScan.Table"        # "crossings"

# stage -> (model kind, PipelinePaths attribute) shown in 3D after the stage
STAGE_OUTPUT = {
    "surfaces": ("surface", "preprocessed"),
    "centerlines": ("centerline", "branch_tree"),
    "anatomy": ("centerline", "anatomical_segments"),
    "surface_labels": ("surface", "branch_labeled_surface"),
}
STRUCTURE_COLOR = {"artery": (0.85, 0.25, 0.25), "veins": (0.25, 0.4, 0.85), "airways": (0.85, 0.85, 0.7)}
# model kind -> (colour, None = the structure's; opacity)
MODEL_STYLE = {
    "surface": (None, 0.6),
    "centerline": ((1.0, 0.9, 0.2), 1.0),
    "lobes": ((0.8, 0.8, 0.8), 0.15),
    "fissures": ((1.0, 0.6, 0.1), 0.7),
}
# Results check box -> model kinds
SHOW_KINDS = ["surfaces", "centerlines", "cutPoints"]  # Results > Show: one check per kind and structure
VISIBILITY_KINDS = {"surfaces": ["surface"], "centerlines": ["centerline"], "lobes": ["lobes", "fissures"],
                    "cutPoints": ["cutPoints"]}


#
# VeScan
#


class VeScan(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("VeScan")
        self.parent.categories = [""]  # top level of the module list, as Welcome and Data
        self.parent.dependencies = []
        self.parent.contributors = ["Edoardo De Rose (KIT, IBT)"]
        self.parent.helpText = _("""
<b>VeScan</b> builds the centerline trees of the pulmonary arteries, veins and airways, labels every branch
with its anatomical segment, finds the vessels that cross the interlobar fissures and cuts the trees to a
number of generations.
<br><br>
<b>Inputs</b> (from <a href="https://github.com/edoardoedr/haversack">haversack</a>):
<ul>
<li>the structure masks (arteries, veins, airways, optional airway wall),</li>
<li>the ranked store <i>lung_vessels.duckn.zip</i>,</li>
<li>the lobe masks (for Preprocess and Anatomy), the CT (optional).</li>
</ul>
<b>Steps</b>: set the inputs, choose structures and stages, Run; then cut the trees in <i>Cut tree</i>
and explore the labels and crossing vessels in <i>Results</i>.
<br><br>
User guide: <a href="https://github.com/edoardoedr/SlicerVeScan#usage">SlicerVeScan</a>.
Methods (Field, Thalweg) and output files: <a href="https://github.com/edoardoedr/VeScan-field">VeScan-field</a>.
""")
        self.parent.acknowledgementText = _("""
Developed at the Institute of Biomedical Engineering (IBT), Karlsruhe Institute of Technology (KIT).
""")


#
# VeScanParameterNode
#


@parameterNodeWrapper
class VeScanParameterNode:
    # Inputs: source ("scene" | "file") + node + path, per input
    ctSource: Annotated[str, Choice(["scene", "file"])] = "scene"
    ctVolume: vtkMRMLScalarVolumeNode
    ctPath: pathlib.Path
    arterySource: Annotated[str, Choice(["scene", "file"])] = "scene"
    veinsSource: Annotated[str, Choice(["scene", "file"])] = "scene"
    airwaysSource: Annotated[str, Choice(["scene", "file"])] = "scene"
    airwayWallSource: Annotated[str, Choice(["scene", "file"])] = "scene"
    lobesSource: Annotated[str, Choice(["scene", "file"])] = "scene"
    arteryPath: pathlib.Path
    veinsPath: pathlib.Path
    airwaysPath: pathlib.Path
    airwayWallPath: pathlib.Path
    lobesPaths: str = ""  # one label map or one file per lobe, separated by ";"
    # scene source of the structures: segmentation + segment ID (synced by hand, see VeScanWidget._onSegmentChanged)
    arterySegmentation: vtkMRMLSegmentationNode
    arterySegmentId: str = ""
    veinsSegmentation: vtkMRMLSegmentationNode
    veinsSegmentId: str = ""
    airwaysSegmentation: vtkMRMLSegmentationNode
    airwaysSegmentId: str = ""
    airwayWallSegmentation: vtkMRMLSegmentationNode
    airwayWallSegmentId: str = ""
    lobesSegmentation: vtkMRMLSegmentationNode
    lobesMapping: str = ""  # lobe table: "RUL=<segment name>;RML=;..." (empty name: None; "": not set yet)
    storePath: pathlib.Path
    # Method
    method: Annotated[str, Choice(["field", "thalweg", "vmtk"])] = "field"
    bifurcations: Annotated[str, Choice(BIFURCATIONS)] = "vmtk"  # method thalweg only
    runArtery: bool = True
    runVeins: bool = True
    runAirways: bool = True
    stagePreprocess: bool = True
    stageCenterlines: bool = True
    stageAnatomy: bool = True
    # Output
    outputDir: pathlib.Path
    # Cut tree
    arteryGenerations: int = 6
    veinsGenerations: int = 6
    airwaysGenerations: int = 6
    capSurface: bool = True
    # lobes where the generations are cut (pipeline.clip.generation_lobes; all = null, as in the config)
    cutLobeRUL: bool = True
    cutLobeRML: bool = True
    cutLobeRLL: bool = True
    cutLobeLUL: bool = True
    cutLobeLLL: bool = True


def isPathSet(path) -> bool:
    # An empty pathlib.Path is "."
    return str(path) not in ("", ".")


def requiredVescanVersion() -> str:
    """vescan-field version required by the module: the tag of the archive in Resources/requirements.txt."""
    return re.search(r"^vescan-field @ \S+/tags/v([\d.]+)\.zip", REQUIREMENTS_PATH.read_text(), re.M).group(1)


def installedVescan():
    """(version, editable) of the installed vescan-field, None if it is not installed."""
    try:
        dist = importlib.metadata.distribution("vescan-field")
    except importlib.metadata.PackageNotFoundError:
        return None
    directUrl = json.loads(dist.read_text("direct_url.json") or "{}")
    return dist.version, directUrl.get("dir_info", {}).get("editable", False)


def ensureVescanField() -> bool:
    """Install vescan-field on first use, or update it when the installed version is older than the required one;
    an editable install (development) is never replaced. Two steps: requirements.txt (vescan-field and its git
    dependencies as GitHub archives) with --no-deps, so that no git is needed and pip never compares two URLs of
    one package; then requirements-deps.txt (their PyPI dependencies). numpy stays at Slicer's version, otherwise
    pip could upgrade it and break Slicer. False if it is missing and the user declines, or if Slicer's Python
    is older than vescan-field's."""
    from packaging.version import Version
    if sys.version_info < MIN_PYTHON:
        slicer.util.errorDisplay(_("VeScan needs Python {required} or newer (3D Slicer 5.12 or newer), this Slicer has "
                                   "Python {python}.").format(required=".".join(map(str, MIN_PYTHON)),
                                                              python=".".join(map(str, sys.version_info[:3]))),
                                 windowTitle="VeScan")
        return False
    installed, required = installedVescan(), requiredVescanVersion()
    if installed is None and importlib.util.find_spec("vescan_field"):
        return True  # importable without package metadata (e.g. on the Python path): left as it is
    if installed and (installed[1] or Version(installed[0]) >= Version(required)):
        return True
    if installed:
        question = _("VeScan needs vescan-field {required}, installed: {installed}. Update it now?").format(
            required=required, installed=installed[0])
    else:
        question = _("VeScan needs the Python package vescan-field {required}, install it now?").format(required=required)
    if not slicer.util.confirmOkCancelDisplay(question, windowTitle="VeScan"):
        return installed is not None  # an older version still runs
    with slicer.util.displayPythonShell(), slicer.util.WaitCursor():
        slicer.util.pip_install(["--upgrade", "--no-deps", "-r", str(REQUIREMENTS_PATH)])
        slicer.util.pip_install(["-r", str(REQUIREMENTS_DEPS_PATH), f"numpy=={np.__version__}"])
    importlib.invalidate_caches()
    if installed and {"vescan_field", "vescan_thalweg"} & set(sys.modules):
        slicer.util.infoDisplay(_("vescan-field updated to {required}: restart Slicer to use it.").format(required=required),
                                windowTitle="VeScan")
    return True


#
# VeScanWidget
#


class VeScanWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/VeScan.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = VeScanLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

        # Path edits and generation sliders are synced by hand: the automatic binding shows an
        # empty path as "." and forces a fixed slider range. Paths use the edit text because
        # currentPathChanged is not emitted for double extensions like "*.nii.gz".
        self._pathEdits = {f"{name}Path": getattr(self.ui, f"{name}PathEdit") for name in PATH_INPUTS}
        self._pathEdits["storePath"] = self.ui.storePathEdit
        self._pathEdits["outputDir"] = self.ui.outputPathEdit
        for paramName, pathEdit in self._pathEdits.items():
            pathEdit.comboBox().editTextChanged.connect(
                lambda path, n=paramName: self._setParameter(n, pathlib.Path(path)))

        # Inputs: a file is loaded and shown as soon as its path is set
        for name in INPUTS:
            self._connectSourceSwitch(name)
        for name in PATH_INPUTS:
            getattr(self.ui, f"{name}PathEdit").comboBox().editTextChanged.connect(
                lambda path, n=name: self._onInputPathChanged(n, path))
        self.ui.lobesPathsEdit.textChanged.connect(self._onLobesPathsChanged)
        self.ui.lobesPathsEdit.editingFinished.connect(self.onLoadLobes)
        self.ui.lobesBrowseButton.clicked.connect(self.onBrowseLobes)
        for name in SEGMENT_INPUTS:
            selector = getattr(self.ui, f"{name}SegmentSelector")
            selector.connect("currentNodeChanged(vtkMRMLNode*)", lambda _node, n=name: self._onSegmentChanged(n))
            selector.connect("currentSegmentChanged(QString)", lambda _id, n=name: self._onSegmentChanged(n))
        self.ui.ctNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda node: self._showSceneNode("ct", node))
        self.ui.lobesNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                          lambda node: self._showSceneNode("lobes", node))
        self.ui.lobesTable.setRowCount(len(LOBES))
        for row, lobe in enumerate(LOBES):
            item = qt.QTableWidgetItem(lobe)
            item.setFlags(qt.Qt.ItemIsEnabled)
            self.ui.lobesTable.setItem(row, 0, item)
        self.ui.lobesTable.hide()
        self.ui.lobesStatusLabel.hide()
        self.ui.loadLobesButton.clicked.connect(self.onLoadLobes)

        # Method
        self.ui.fieldRadio.toggled.connect(self.onMethodChanged)
        self.ui.thalwegRadio.toggled.connect(self.onMethodChanged)
        self.ui.bifurcationsCombo.currentIndexChanged.connect(
            lambda index: self._setParameter("bifurcations", BIFURCATIONS[index]))

        # Output
        self.ui.runButton.clicked.connect(self.onRun)
        self.ui.cancelButton.clicked.connect(self.onCancel)

        # Cut tree
        self.ui.clipCollapsibleButton.hide()
        for name in STRUCTURES:
            getattr(self.ui, f"{name}GenerationsSlider").valueChanged.connect(
                lambda value, n=name: self.onGenerationsChanged(n, value))
        for lobe in LOBES:
            getattr(self.ui, f"cutLobe{lobe}Check").toggled.connect(self.onCutLobesChanged)
        self.ui.previewCheck.toggled.connect(self.onPreviewToggled)
        self.ui.applyCutButton.clicked.connect(self.onApplyCut)

        # Results
        for kind, structure, check in self._showChecks():
            check.toggled.connect(lambda visible, k=kind, s=structure: self.onDisplayChanged(k, visible, s))
        self.ui.colorByCombo.currentIndexChanged.connect(self.onColorByChanged)
        self.ui.lobesColorByCombo.currentIndexChanged.connect(self.onLobesColorByChanged)
        self.ui.showLegendsCheck.toggled.connect(self.onShowLegendsToggled)
        self.ui.crossingsTableView.clicked.connect(lambda index: self.onCrossingSelected(index.row()))
        self.ui.exportStatsButton.clicked.connect(self.onExportStatistics)

        self.restoreFromScene()  # also when the module is opened on a scene loaded before

    def cleanup(self) -> None:
        self.removeObservers()
        self.logic.removeTemporaryOutput()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateRunButtonState)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)
        self.logic.removeTemporaryOutput()

    def onSceneEndClose(self, caller, event) -> None:
        self.restoreFromScene()  # empty scene: hides the lobe table, Cut tree and Results

    def onSceneEndImport(self, caller, event) -> None:
        self.restoreFromScene()

    def restoreFromScene(self) -> None:
        """Lobe table, Cut tree and Results from the nodes the module created (saved with the scene)."""
        # the logic first: the paths set in the GUI then find their nodes instead of loading the files again
        outputDir = self.logic.getParameterNode().outputDir
        self.logic.restoreFromScene(pathlib.Path(outputDir) if isPathSet(outputDir) else None)
        self.initializeParameterNode()
        p = self._parameterNode
        lobesLoaded = self.logic.lobesNode or (p.lobesSource == "scene" and p.lobesSegmentation)
        if lobesLoaded and importlib.util.find_spec("vescan_field"):
            self.onLoadLobes()  # file lobes: the node in the scene, not loaded again
        else:
            self.ui.lobesTable.hide()
            self.ui.lobesStatusLabel.hide()
        self._onCenterlinesDone(keepGenerations=True)
        if self.logic.crossingsTableNode or self.logic.hasResults():
            self._showResults()
        else:
            self.ui.resultsCollapsibleButton.enabled = False

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: VeScanParameterNode | None) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateRunButtonState)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateRunButtonState)
            # Restore what is not bound automatically
            for paramName, pathEdit in self._pathEdits.items():
                path = getattr(self._parameterNode, paramName)
                pathEdit.currentPath = str(path) if isPathSet(path) else ""
            self.ui.lobesPathsEdit.text = self._parameterNode.lobesPaths
            for name in SEGMENT_INPUTS:
                # read both first: setting the node selects its first segment and updates the parameters
                segmentation = getattr(self._parameterNode, f"{name}Segmentation")
                segmentId = getattr(self._parameterNode, f"{name}SegmentId")
                selector = getattr(self.ui, f"{name}SegmentSelector")
                selector.setCurrentNode(segmentation)
                if segmentId:
                    selector.setCurrentSegmentID(segmentId)
            for name in INPUTS:
                isFile = getattr(self._parameterNode, f"{name}Source") == "file"
                getattr(self.ui, f"{name}FileRadio" if isFile else f"{name}SceneRadio").checked = True
            # read both first: checking a radio updates the parameters
            method, bifurcations = self._parameterNode.method, self._parameterNode.bifurcations
            getattr(self.ui, f"{method if method in METHOD_PACKAGE else 'field'}Radio").checked = True
            self.ui.bifurcationsCombo.currentIndex = BIFURCATIONS.index(bifurcations)
            self.onMethodChanged()
            self._updateRunButtonState()

    def _setParameter(self, name, value) -> None:
        if self._parameterNode:
            setattr(self._parameterNode, name, value)

    #
    # Inputs
    #

    def _connectSourceSwitch(self, name) -> None:
        stack = getattr(self.ui, f"{name}Stack")
        getattr(self.ui, f"{name}FileRadio").toggled.connect(lambda isFile: self._onSourceToggled(name, stack, isFile))
        self._onSourceToggled(name, stack, False)

    def _onSourceToggled(self, name, stack, isFile) -> None:
        stack.currentIndex = 1 if isFile else 0
        # Size the stack to the visible page only
        for i in range(stack.count):
            policy = qt.QSizePolicy.Preferred if i == stack.currentIndex else qt.QSizePolicy.Ignored
            stack.widget(i).setSizePolicy(qt.QSizePolicy.Preferred, policy)
        self._setParameter(f"{name}Source", "file" if isFile else "scene")
        if name in SEGMENT_INPUTS:
            self._showSceneSegments()

    def _isInputSet(self, name) -> bool:
        p = self._parameterNode
        if name == "lobes":
            return bool(self._lobesPaths()) if p.lobesSource == "file" else p.lobesSegmentation is not None
        if getattr(p, f"{name}Source") == "file":
            return isPathSet(getattr(p, f"{name}Path"))
        if name == "ct":
            return p.ctVolume is not None
        return getattr(p, f"{name}Segmentation") is not None and bool(getattr(p, f"{name}SegmentId"))

    def _onSegmentChanged(self, name) -> None:
        selector = getattr(self.ui, f"{name}SegmentSelector")
        segmentation = selector.currentNode()
        if segmentation and not selector.currentSegmentID() and segmentation.GetSegmentation().GetNumberOfSegments():
            selector.setCurrentSegmentID(segmentation.GetSegmentation().GetNthSegmentID(0))  # calls this again
            return
        self._setParameter(f"{name}Segmentation", selector.currentNode())
        self._setParameter(f"{name}SegmentId", selector.currentSegmentID() or "")
        self._showSceneSegments()

    def _showSceneSegments(self) -> None:
        p = self._parameterNode
        if p:
            self.logic.showSceneSegments([(getattr(p, f"{name}Segmentation"), getattr(p, f"{name}SegmentId"))
                                          for name in SEGMENT_INPUTS if getattr(p, f"{name}Source") == "scene"])

    def _showSceneNode(self, name, node) -> None:
        """CT (slice views) or lobe segmentation (3D) chosen in the scene."""
        p = self._parameterNode
        if p and node and getattr(p, f"{name}Source") == "scene":
            with slicer.util.WaitCursor():
                self.logic.showInput(name, node)

    def _onInputPathChanged(self, name, path) -> None:
        """Load and show a file input as soon as its path points to an existing file."""
        if getattr(self._parameterNode, f"{name}Source", None) != "file" or not os.path.isfile(path):
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to load {path}.").format(path=path), waitCursor=True):
            if name == "ct":
                self.logic.loadCT("file", path=path)
            else:
                self.logic.loadStructure(name, "file", path=path)

    def _lobesPaths(self) -> list:
        return [p.strip() for p in self._parameterNode.lobesPaths.split(";") if p.strip()]

    def onBrowseLobes(self) -> None:
        paths = qt.QFileDialog.getOpenFileNames(slicer.util.mainWindow(), _("Lobes: select all 5 lobe files, or one label map"),
                                                "", SEGMENTATION_FILES)
        if paths:
            self.ui.lobesPathsEdit.text = "; ".join(paths)
            self.onLoadLobes()

    def _structureInput(self, name):
        p = self._parameterNode
        return self.logic.loadStructure(name, getattr(p, f"{name}Source"), getattr(p, f"{name}Segmentation"),
                                        getattr(p, f"{name}SegmentId"), getattr(p, f"{name}Path"))

    def _onLobesPathsChanged(self, text) -> None:
        if self._parameterNode and text != self._parameterNode.lobesPaths:
            self._parameterNode.lobesMapping = ""  # new files: the table is mapped again automatically
            self._parameterNode.lobesPaths = text

    def _savedLobesMapping(self, names) -> dict:
        """lobe -> segment name ("" = None) saved in the parameter node; {} if not saved or a segment is gone."""
        mapping = dict(item.split("=", 1) for item in self._parameterNode.lobesMapping.split(";") if "=" in item)
        valid = set(mapping) == set(LOBES) and all(not name or name in names for name in mapping.values())
        return mapping if valid else {}

    def onLoadLobes(self) -> None:
        """Load the lobes and fill the table: as saved in the parameter node, else automatically: a segment
        named as the lobe or as its TotalSegmentator mask (e.g. lung_upper_lobe_right), 5 segments with other
        names in order, any other lobe None."""
        p = self._parameterNode
        if (p.lobesSource == "file" and not self._lobesPaths()) or not ensureVescanField():
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to load lobes."), waitCursor=True):
            from vescan_field.lobes import LOBE_SEGMENTATION_NAME
            names = self.logic.loadLobes(p.lobesSource, p.lobesSegmentation, self._lobesPaths())
            known = {lobe: [n for n in (lobe, LOBE_SEGMENTATION_NAME[lobe]) if n in names] for lobe in LOBES}
            byOrder = len(names) == len(LOBES) and not any(known.values())
            saved = self._savedLobesMapping(names)
            for row, lobe in enumerate(LOBES):
                combo = qt.QComboBox()
                combo.addItems([_("None")] + names)
                if saved:
                    combo.currentIndex = names.index(saved[lobe]) + 1 if saved[lobe] else 0
                elif known[lobe]:
                    combo.currentIndex = names.index(known[lobe][0]) + 1
                else:
                    combo.currentIndex = row + 1 if byOrder else 0
                combo.currentIndexChanged.connect(self.onLobeMappingChanged)
                self.ui.lobesTable.setCellWidget(row, 1, combo)
            self.ui.lobesTable.show()
            self.onLobeMappingChanged()

    def onLobeMappingChanged(self) -> None:
        combos = [self.ui.lobesTable.cellWidget(row, 1) for row in range(len(LOBES))]
        self._setParameter("lobesMapping", ";".join(f"{lobe}={combo.currentText if combo.currentIndex else ''}"
                                                    for lobe, combo in zip(LOBES, combos)))
        self.logic.colorLobes(self._lobeSegments())
        missing = self._missingLobes()
        if len(missing) == len(LOBES):
            text, color = _("No lobe set: load the lobe files (all 5 together), or pick their labels."), "#c0392b"
        elif missing:
            text, color = _("{n}/5 lobes, without {lobes}: VeScan runs on the set lobes only. "
                            "For all lobes load the 5 files together, or pick their labels.").format(
                                n=len(LOBES) - len(missing), lobes=", ".join(missing)), "#d35400"
        else:
            text, color = _("All 5 lobes set."), ""
        self.ui.lobesStatusLabel.text = text
        self.ui.lobesStatusLabel.styleSheet = f"color: {color}" if color else ""
        self.ui.lobesStatusLabel.show()
        self._updateRunButtonState()

    def _lobeSegments(self) -> dict:
        """lobe -> segment ID in the lobe segmentation (None: not set), from the lobe table."""
        if self.ui.lobesTable.isHidden():
            self.onLoadLobes()
        segmentation = self.logic.lobesNode.GetSegmentation()
        combos = [self.ui.lobesTable.cellWidget(row, 1) for row in range(len(LOBES))]
        return {lobe: segmentation.GetSegmentIdBySegmentName(combo.currentText) if combo.currentIndex else None
                for lobe, combo in zip(LOBES, combos)}

    def _missingLobes(self) -> list:
        """Lobes without a segment in the lobe table (none before the lobes are loaded)."""
        if self.ui.lobesTable.isHidden():
            return []
        return [LOBES[row] for row in range(len(LOBES)) if not self.ui.lobesTable.cellWidget(row, 1).currentIndex]

    #
    # Method / Output
    #

    def onMethodChanged(self) -> None:
        method = "field" if self.ui.fieldRadio.checked else "thalweg" if self.ui.thalwegRadio.checked else "vmtk"
        self._setParameter("method", method)
        self.ui.bifurcationsLabel.visible = self.ui.bifurcationsCombo.visible = method == "thalweg"

    def _updateRunButtonState(self, caller=None, event=None) -> None:
        p = self._parameterNode
        if not p:
            return
        structures = self._runStructures()
        missing = []
        if not structures:
            missing.append(_("a structure"))
        missing += [name for name in structures if not self._isInputSet(name)]  # CT optional
        if p.method not in METHOD_PACKAGE:
            missing.append(_("an available method"))
        elif not isPathSet(p.storePath):
            missing.append(_("field store"))
        if (p.stagePreprocess or p.stageAnatomy) and not self._isInputSet("lobes"):
            missing.append(_("lobes"))
        elif (p.stagePreprocess or p.stageAnatomy) and len(self._missingLobes()) == len(LOBES):
            missing.append(_("a lobe"))
        if not isPathSet(p.outputDir) and not (p.stagePreprocess and p.stageCenterlines):
            missing.append(_("output folder (with stages off, the results of a previous run are read from it)"))
        self.ui.runButton.enabled = not missing and not self.ui.cancelButton.enabled
        self.ui.runButton.toolTip = _("Missing: ") + ", ".join(missing) if missing else _("Run VeScan")

    def onRun(self) -> None:
        if not ensureVescanField():
            return
        p = self._parameterNode
        if not importlib.util.find_spec(METHOD_PACKAGE[p.method]):
            slicer.util.errorDisplay(_("The installed vescan-field has no {package}: update it to {required} "
                                       "(an editable install: pull the clone and install it again).").format(
                                         package=METHOD_PACKAGE[p.method], required=requiredVescanVersion()),
                                     windowTitle="VeScan")
            return
        self.ui.logText.clear()
        self.ui.cancelButton.enabled = True
        self.ui.runButton.enabled = False
        try:
            with slicer.util.tryWithErrorDisplay(_("VeScan failed."), waitCursor=True):
                ctNode = self.logic.loadCT(p.ctSource, p.ctVolume, p.ctPath) if self._isInputSet("ct") else None
                structures = {name: [self._structureInput(name)] for name in self._runStructures()}
                if "airways" in structures and self._isInputSet("airwayWall"):
                    structures["airways"].append(self._structureInput("airwayWall"))
                lobeSegments = self._lobeSegments() if p.stagePreprocess or p.stageAnatomy else {}
                finished = self.logic.run(p, ctNode, structures, lobeSegments, self._onProgress,
                                          self.ui.logText.appendPlainText)
                if finished:
                    self._onCenterlinesDone()
                    self._onRunFinished()
        finally:
            self.ui.cancelButton.enabled = False
            self._updateRunButtonState()

    def onCancel(self) -> None:
        self.logic.cancel()

    def _onProgress(self, stage, structure, i, n) -> None:
        self.ui.progressBar.maximum = n
        self.ui.progressBar.value = i
        self.ui.statusLabel.text = f"{stage}: {structure} ({i}/{n})" if structure else f"{stage} ({i}/{n})"

    def _runStructures(self):
        return [s for s in STRUCTURES if getattr(self._parameterNode, f"run{s.capitalize()}")]

    #
    # Cut tree
    #

    def _onCenterlinesDone(self, keepGenerations=False) -> None:
        """Cut tree for the structures with a centerline: sliders on its generation range (at the maximum
        after a run, at the saved value when a scene is loaded); Apply cut needs the run's files."""
        withTree = [s for s in STRUCTURES if self.logic.hasCenterline(s)]
        for name in STRUCTURES:
            visible = name in withTree
            getattr(self.ui, f"{name}GenerationsLabel").visible = visible
            slider = getattr(self.ui, f"{name}GenerationsSlider")
            slider.visible = visible
            if visible:
                saved = getattr(self._parameterNode, f"{name}Generations")
                slider.minimum, slider.maximum = self.logic.getGenerationRange(name)
                slider.value = min(max(saved, slider.minimum), slider.maximum) if keepGenerations else slider.maximum
        self.ui.clipCollapsibleButton.visible = bool(withTree)
        self.ui.clipCollapsibleButton.collapsed = False
        # lobe checks: only the lobes of the run, and only with the anatomy (Reaches_<LOBE> arrays)
        byLobe = any(self.logic.canCutByLobe(s) for s in withTree)
        runLobes = self.logic.runLobes()
        self.ui.cutLobesLabel.enabled = byLobe
        for lobe in LOBES:
            getattr(self.ui, f"cutLobe{lobe}Check").enabled = byLobe and lobe in runLobes
        self._updateApplyCut()

    def _cutLobes(self):
        """pipeline.clip.generation_lobes from the lobe checks (None: all the lobes of the run; []: none)."""
        # from the checks: toggled reaches this widget before connectGui updates the parameter node
        return self.logic.generationLobes([lobe for lobe in LOBES if getattr(self.ui, f"cutLobe{lobe}Check").checked])

    def _updateApplyCut(self) -> None:
        withTree = [s for s in STRUCTURES if self.logic.hasCenterline(s)]
        canCut = any(self.logic.canCut(s) for s in withTree)
        noLobe = canCut and any(self.logic.canCutByLobe(s) for s in withTree) and self._cutLobes() == []
        self.ui.applyCutButton.enabled = canCut and not noLobe
        self.ui.applyCutButton.toolTip = (
            "" if canCut and not noLobe else _("Check at least one lobe to cut.") if noLobe else
            _("The files of the run are not in the output folder anymore (temporary folder, or moved): run again to cut."))

    def onGenerationsChanged(self, structure, value) -> None:
        self._setParameter(f"{structure}Generations", int(value))
        if self.ui.previewCheck.checked:
            self.logic.previewCut(structure, int(value), self._cutLobes())

    def onCutLobesChanged(self) -> None:
        if not self._parameterNode:
            return
        self._updateApplyCut()
        if self.ui.previewCheck.checked:
            self.onPreviewToggled(True)

    def onPreviewToggled(self, enabled) -> None:
        for name in STRUCTURES:
            if self.logic.hasCenterline(name):
                generations = getattr(self._parameterNode, f"{name}Generations") if enabled else None
                self.logic.previewCut(name, generations, self._cutLobes())
        if not enabled:
            self.logic.setColorBy(*COLOR_BY[self.ui.colorByCombo.currentIndex])

    def onApplyCut(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to cut the tree."), waitCursor=True):
            for name in STRUCTURES:
                if self.logic.canCut(name):
                    self.logic.applyCut(name, getattr(self._parameterNode, f"{name}Generations"),
                                        self._parameterNode.capSurface, self._cutLobes())
            for kind, structure, check in self._showChecks():
                if kind == "cutPoints":
                    self.onDisplayChanged(kind, check.checked, structure)

    #
    # Results
    #

    def _showChecks(self):
        """(kind, structure, check) of Results > Show; lobes and fissures are one model for all structures."""
        checks = [(kind, structure, getattr(self.ui, f"show{kind[0].upper()}{kind[1:]}{structure.capitalize()}Check"))
                  for kind in SHOW_KINDS for structure in STRUCTURES]
        return checks + [("lobes", None, self.ui.showLobesCheck)]

    def _onRunFinished(self) -> None:
        self.logic.loadResults()
        self._showResults()
        for kind, structure, check in self._showChecks():
            self.onDisplayChanged(kind, check.checked, structure)

    def _showResults(self) -> None:
        self.ui.crossingsTableView.setMRMLTableNode(self.logic.crossingsTableNode)
        self.ui.resultsCollapsibleButton.enabled = True
        self.ui.resultsCollapsibleButton.collapsed = False
        for kind, structure, check in self._showChecks():  # a structure without results: its column disabled
            check.enabled = structure is None or self.logic.hasResults(structure)
        self.logic.updateLegends()

    def onDisplayChanged(self, kind, visible, structure=None) -> None:
        self.logic.setVisibility(kind, visible, structure)

    def onColorByChanged(self, index) -> None:
        self.ui.previewCheck.checked = False  # the preview colours by Generation
        self.logic.setColorBy(*COLOR_BY[index])

    def onShowLegendsToggled(self, visible) -> None:
        self.logic.showLegends = visible
        self.logic.updateLegends()

    def onLobesColorByChanged(self, index) -> None:
        self.logic.setLobesColorBy(*LOBES_COLOR_BY[index])

    def onCrossingSelected(self, row) -> None:
        tableNode = self.ui.crossingsTableView.mrmlTableNode()
        if tableNode and row < tableNode.GetNumberOfRows():
            self.logic.focusCrossing(tableNode.GetCellText(row, 1), int(tableNode.GetCellText(row, 0)))

    def onExportStatistics(self) -> None:
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), _("Export statistics"), "", "CSV (*.csv)")
        if path:
            self.logic.exportStatistics(path)


#
# VeScanLogic
#


class RunCancelled(Exception):
    pass


class UiLogHandler(logging.Handler):
    """Forwards the vescan_field / vescan_thalweg messages to the module log and keeps the GUI responsive."""

    def __init__(self, callback, package):
        super().__init__()
        self.callback = callback
        self.prefix = f"{package}."

    def emit(self, record):
        self.callback(f"{record.name.removeprefix(self.prefix)}: {record.getMessage()}")
        slicer.app.processEvents()


class VeScanLogic(ScriptedLoadableModuleLogic):
    """Inputs are loaded with Slicer, vescan_field or vescan_thalweg (by method) computes and saves (VeScan's
    output layout, the same for both), and each output is shown in 3D as soon as it is written."""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self._cancelRequested = False
        self.package = METHOD_PACKAGE["field"]  # package of the last run (its clip stage cuts the tree)
        self._inputs = {}          # input name -> (file path(s), node loaded from them)
        self._models = {}          # (kind, structure) -> displayed node
        self.lobesNode = None
        self.temporaryOutput = None  # output folder of a run without output folder (removed later)
        self._sceneVisibility = {}     # segmentation node ID -> {segment ID: 3D visibility before the module}
        self.contexts = {}         # structure -> vescan StructureContext of the last run
        self.statisticsRows = []
        self.crossingsTableNode = None
        self.colorBy = COLOR_BY[0]      # (array, colour table) of the surfaces and centerlines
        self.lobesColorBy = LOBES_COLOR_BY[0]  # (array, values) of the lobe mesh
        self.showLegends = True

    def getParameterNode(self):
        return VeScanParameterNode(super().getParameterNode())

    def module(self, name, package=None):
        """Module `name` of the method's package (default: the last run's): e.g. module("stages.clip")."""
        return importlib.import_module(f"{package or self.package}.{name}")

    # Inputs

    def _loadInput(self, name, key, load):
        """Node of a file input: loaded and shown once, replaced when its path changes."""
        oldKey, old = self._inputs.get(name, (None, None))
        if old and slicer.mrmlScene.IsNodePresent(old):
            if oldKey == key:
                return old
            slicer.mrmlScene.RemoveNode(old)
        node = load()
        node.SetAttribute(INPUT_ATTRIBUTE, name)
        node.SetAttribute(INPUT_KEY_ATTRIBUTE, json.dumps(key))
        self._inputs[name] = (key, node)
        self.showInput(name, node)
        return node

    def showInput(self, name, node):
        """CT in the slice views, a segmentation in 3D (structures in their colour)."""
        if node.IsA("vtkMRMLScalarVolumeNode"):
            slicer.util.setSliceViewerLayers(background=node, fit=True)
            return
        node.CreateClosedSurfaceRepresentation()
        color = STRUCTURE_COLOR.get("airways" if name == "airwayWall" else name)
        segmentation = node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            if color:
                segmentation.GetNthSegment(i).SetColor(*color)
        node.GetDisplayNode().SetOpacity3D(0.3 if name == "lobes" else 1.0)
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()

    def showSceneSegments(self, selections):
        """Structures chosen in the scene, in 3D: selections = [(segmentation, segment ID), ...]. Of these
        segmentations only the chosen segments are shown in 3D (still all in the slice views), colours unchanged;
        a segmentation no longer chosen gets back its segments' 3D visibility."""
        chosen = {}
        for segmentation, segmentId in selections:
            if segmentation and segmentId:
                chosen.setdefault(segmentation.GetID(), set()).add(segmentId)
        for nodeId in list(self._sceneVisibility):
            node = slicer.mrmlScene.GetNodeByID(nodeId)
            if nodeId not in chosen and node and node.GetDisplayNode():
                for segmentId, visible in self._sceneVisibility[nodeId].items():
                    node.GetDisplayNode().SetSegmentVisibility3D(segmentId, visible)
            if nodeId not in chosen:
                del self._sceneVisibility[nodeId]
        for nodeId, segmentIds in chosen.items():
            node = slicer.mrmlScene.GetNodeByID(nodeId)
            if not node.GetDisplayNode():
                node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            allIds = [node.GetSegmentation().GetNthSegmentID(i) for i in range(node.GetSegmentation().GetNumberOfSegments())]
            if nodeId not in self._sceneVisibility:
                self._sceneVisibility[nodeId] = {i: display.GetSegmentVisibility3D(i) for i in allIds}
                with slicer.util.WaitCursor():
                    node.CreateClosedSurfaceRepresentation()  # all its segments, once
                slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
            display.SetVisibility(True)
            display.SetVisibility3D(True)
            for i in allIds:
                display.SetSegmentVisibility3D(i, i in segmentIds)

    def loadCT(self, source, node=None, path=None):
        if source == "scene":
            return node
        return self._loadInput("ct", str(path), lambda: slicer.util.loadVolume(str(path)))

    def loadStructure(self, name, source, segmentation=None, segmentId=None, path=None):
        """(segmentation node, segment ID) of a structure mask."""
        if source == "file":
            segmentation = self._loadInput(name, str(path), lambda: slicer.util.loadSegmentation(str(path)))
            segmentId = segmentation.GetSegmentation().GetNthSegmentID(0)
        if not segmentation or not segmentId:
            raise ValueError(f"No segment selected for {name}")
        return segmentation, segmentId

    def loadStore(self, path):
        """The ranked store as the method reads it: vescan_field's Store, or thalweg's FieldStore."""
        if self.package == METHOD_PACKAGE["thalweg"]:
            from thalweg.store import open_store
            return open_store(path)
        from vescan_field.store import Store
        return Store(path)

    def loadLobes(self, source, segmentation=None, paths=()):
        """Segment names of the lobe segmentation. From files: one segmentation "Lobes" with a segment
        per label of a label map, or one segment per file named as the file (e.g. lung_upper_lobe_right)."""
        if source == "scene":
            self.lobesNode = segmentation
        else:
            self.lobesNode = self._loadInput("lobes", tuple(map(str, paths)), lambda: self._mergeSegmentations(paths))
        segments = self.lobesNode.GetSegmentation()
        return [segments.GetNthSegment(i).GetName() for i in range(segments.GetNumberOfSegments())]

    def colorLobes(self, lobeSegments):
        """Lobe segments in the colours of VeScan's LobeLabel table (value = position in LOBE_ORDER + 1)."""
        from vescan_field.lobes import LOBE_ORDER
        table, rgba = self.colorTable("ColorTableLobeLabel_EN"), [0.0] * 4
        for lobe, segmentId in lobeSegments.items():
            segment = segmentId and self.lobesNode.GetSegmentation().GetSegment(segmentId)
            if segment and table.GetColor(LOBE_ORDER.index(lobe) + 1, rgba):
                segment.SetColor(*rgba[:3])

    @staticmethod
    def _mergeSegmentations(paths):
        merged = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Lobes")
        merged.CreateDefaultDisplayNodes()
        for path in paths:
            loaded = slicer.util.loadSegmentation(str(path))
            segments = loaded.GetSegmentation()
            if segments.GetNumberOfSegments() == 1:
                segments.GetNthSegment(0).SetName(pathlib.Path(path).name.split(".")[0])
            for i in range(segments.GetNumberOfSegments()):
                merged.GetSegmentation().CopySegmentFromSegmentation(segments, segments.GetNthSegmentID(i))
            slicer.mrmlScene.RemoveNode(loaded)
        return merged

    @staticmethod
    def geometry(volumeNode, package=METHOD_PACKAGE["field"]):
        """vescan Geometry (of `package`) of a volume node: LPS, as SimpleITK reads the same image."""
        Geometry = importlib.import_module(f"{package}.field").Geometry
        m = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASDirectionMatrix(m)
        directions = np.array([[m.GetElement(r, c) for c in range(3)] for r in range(3)])
        return Geometry(np.array(volumeNode.GetSpacing())[::-1], RAS_TO_LPS @ np.array(volumeNode.GetOrigin()),
                        RAS_TO_LPS @ directions)

    @staticmethod
    def mask(segmentation, segmentId, referenceNode):
        """Segment as a boolean array (z,y,x) on the reference grid."""
        return slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, referenceNode) > 0

    @staticmethod
    def referenceGrid(segmentation, segmentId):
        """Without a CT: a label map on the segmentation's full grid (the grid of the mask file, which is
        what vescan uses as CT geometry)."""
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "VeScan reference grid")
        segmentIds = vtk.vtkStringArray()
        segmentIds.InsertNextValue(segmentId)
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentation, segmentIds, labelmap, None, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
        return labelmap

    def buildConfig(self, params, outputDir, airwayWall=False, lobes=None):
        """The method's bundled JSON with the keys set by the UI, saved in the output folder and validated by its
        package. lobes: the active lobes (pipeline.lobes.enabled; None: as in the JSON)."""
        package = METHOD_PACKAGE[params.method]
        config, io = self.module("config", package), self.module("io", package)
        data = json.loads(CONFIG_PATHS[params.method].read_text())
        batch = data["batch"]
        batch.update(patients_root_dir=str(outputDir), surfaces_root_dir=str(outputDir / "surfaces"),
                     results_root_dir=str(outputDir), store=pathlib.Path(params.storePath).name)
        for name in STRUCTURES:
            batch["structures"][name]["enabled"] = getattr(params, f"run{name.capitalize()}")
        if not airwayWall:
            airways = batch["structures"]["airways"]
            airways["classes"] = [c for c in airways["classes"] if c != AIRWAY_WALL_CLASS]
        data["pipeline"]["stages"] = dict(surfaces=params.stagePreprocess, centerlines=params.stageCenterlines,
                                          anatomy=params.stageAnatomy, surface_labels=True, clip=False)
        data["pipeline"]["clip"]["cap"] = params.capSurface
        if params.method == "thalweg":
            data["pipeline"]["centerlines"]["bifurcations"] = params.bifurcations
        if lobes is not None:
            data["pipeline"]["lobes"]["enabled"] = lobes
        outputDir.mkdir(parents=True, exist_ok=True)
        io.write_json(outputDir / "vescan_config.json", data)
        return config.build(data, base_dir=outputDir)

    # Run

    def run(self, params, ctNode, structures, lobeSegments, progressCallback, logCallback=None) -> bool:
        """ctNode: reference grid, optional (None: the grid of the first structure's mask);
        structures: name -> [(segmentation, segment ID), ...] (airways: + airway wall);
        lobeSegments: lobe -> segment ID in self.lobesNode (None: lobe excluded, pipeline.lobes.enabled).
        Returns False if cancelled."""
        self.package = METHOD_PACKAGE[params.method]
        io, log = self.module("io"), self.module("log")
        lobe_mask_names = self.module("lobes").lobe_mask_names
        STAGES = self.module("orchestrator").STAGES
        SurfacePaths = self.module("paths").SurfacePaths

        self._cancelRequested = False
        self.contexts = {}
        self.removeTemporaryOutput()
        if isPathSet(params.outputDir):
            outputDir = pathlib.Path(params.outputDir)
        else:
            outputDir = self.temporaryOutput = pathlib.Path(slicer.app.temporaryPath) / "VeScan" / time.strftime("%Y%m%d-%H%M%S")
            if logCallback:
                logCallback(_("No output folder: working in {path} (removed on scene close or with the next run; "
                              "save the scene to keep the results).").format(path=outputDir))
        cfg = self.buildConfig(params, outputDir, airwayWall=len(structures.get("airways", [])) > 1,
                               lobes=[lobe for lobe, segmentId in lobeSegments.items() if segmentId] if lobeSegments else None)
        stages = [name for name in STAGES if getattr(cfg.pipeline.stages, name)]
        enabled = cfg.enabled_structures()
        # the VeScan models replace the input segmentations in 3D (still shown in the slice views)
        for segmentation in {seg for inputs in structures.values() for seg, _ in inputs} | {self.lobesNode} - {None}:
            segmentation.GetDisplayNode().SetVisibility3D(False)
        steps = int(cfg.pipeline.stages.surfaces) + len(enabled) * (1 + len(stages))
        done = 0

        def begin(stage, structure=""):
            nonlocal done
            if self._cancelRequested:
                raise RunCancelled()
            progressCallback(stage, structure, done, steps)
            slicer.app.processEvents()
            done += 1

        log.setup(cfg.logging.level)
        vescanLogger = logging.getLogger(log.ROOT)
        handler = UiLogHandler(logCallback, self.package) if logCallback else None
        if handler:
            vescanLogger.addHandler(handler)
        store = self.loadStore(params.storePath)
        firstSegmentation, firstSegmentId = next(iter(structures.values()))[0]
        reference = ctNode or self.referenceGrid(firstSegmentation, firstSegmentId)
        patient = ctNode.GetName() if ctNode else outputDir.name  # vescan: the patient folder's name
        fileLog = log.to_file(outputDir / cfg.logging.file_name) if cfg.logging.file_name else contextlib.nullcontext()
        try:
            with fileLog:
                ctGeom = self.geometry(reference, self.package)
                lobes = lobe_mask_names(cfg.pipeline.lobes.enabled, cfg.pipeline.lobes.names)
                surfaces = SurfacePaths(cfg.batch.surfaces_root_dir)
                if cfg.pipeline.stages.surfaces:
                    begin("lobes")
                    self._runLobes(cfg, lobes, lobeSegments, reference, ctGeom, surfaces)
                for structure, classes in enabled.items():
                    begin("margin", structure)
                    mask = np.logical_or.reduce([self.mask(seg, sid, reference) for seg, sid in structures[structure]])
                    ctx = self._buildContext(cfg, patient, structure, classes, store, mask, ctGeom,
                                             lobes, surfaces, outputDir)
                    self.contexts[structure] = ctx
                    summary = self._summary(cfg, ctx, store)
                    for name in stages:
                        begin(name, structure)
                        stage = self.module(f"stages.{name}")
                        summary[name] = stage.run(ctx, getattr(cfg.pipeline, name))
                        self._showStageOutput(name, ctx)
                    summary["timings_s"] = ctx.timings
                    io.write_json(ctx.paths.summary, summary)
            progressCallback("done", "", steps, steps)
            return True
        except RunCancelled:
            vescanLogger.warning("cancelled")
            return False
        finally:
            if hasattr(store, "close"):
                store.close()
            if reference is not ctNode:
                slicer.mrmlScene.RemoveNode(reference)
            if handler:
                vescanLogger.removeHandler(handler)

    def removeTemporaryOutput(self):
        """Remove the temporary output folder of a run without output folder."""
        if self.temporaryOutput:
            shutil.rmtree(self.temporaryOutput, ignore_errors=True)
            self.temporaryOutput = None

    def _runLobes(self, cfg, lobes, lobeSegments, referenceNode, ctGeom, surfaces):
        """As vescan's stages.surfaces.run_patient, with the lobe masks from the lobe segmentation:
        one surface per lobe (read again by anatomy), lobe_segments.vtk, lobe_fissures.vtk and the legend."""
        io = self.module("io")
        lobe_segments = self.module("stages.surfaces.lobe_segments")
        convert_mask = self.module("stages.surfaces.convert").convert_mask
        save_surface = self.module("surface").save_surface

        surfaces.surfaces_dir.mkdir(parents=True, exist_ok=True)
        c = cfg.pipeline.surfaces.convert
        polys = {}
        for lobe, maskName in lobes.items():
            poly = convert_mask(self.mask(self.lobesNode, lobeSegments[lobe], referenceNode), ctGeom, method=c.method,
                                smoothing_factor=c.smoothing_factor, decimation_factor=c.decimation_factor,
                                compute_normals=c.compute_normals)
            if poly is None:
                logging.getLogger(f"{self.package}.slicer").warning(f"empty lobe {lobe}")
                continue
            save_surface(poly, surfaces.surface(maskName))
            polys[lobe] = poly
        l = cfg.pipeline.surfaces.lobe_segments
        if not (l.enabled and polys):
            return
        combined, fissures = lobe_segments.build_combined_lobes(
            polys, extract_fissures_too=l.extract_fissures, tolerance=l.fissure_contact_tolerance,
            min_component_area=l.min_fissure_component_area, min_fissure_area=l.min_fissure_area)
        save_surface(combined, surfaces.segments)
        self._showModel("lobes", "", surfaces.segments)
        sheets = lobe_segments.build_fissure_surface(fissures)
        if sheets is not None:
            save_surface(sheets, surfaces.fissures)
            self._showModel("fissures", "", surfaces.fissures)
        io.write_json(surfaces.segments_legend, lobe_segments.legend(fissures))

    def _buildContext(self, cfg, patient, structure, classes, store, mask, ctGeom, lobes, surfaces, outputDir):
        """As the package's orchestrator.build_context, with the mask from Slicer. Field: the margin resampled on
        the CT grid around the mask; thalweg: the union margin on the store's own grid."""
        pathsmod = self.module("paths")
        paths = pathsmod.PipelinePaths(pathsmod.final_export_dir(outputDir, structure), debug=cfg.pipeline.debug_files)
        paths.make_dirs()
        common = dict(patient=patient, structure=structure, classes=classes, store=store, patient_dir=outputDir,
                      results_dir=outputDir, surfaces=surfaces, paths=paths, mask=mask, ct_geom=ctGeom, lobes=lobes,
                      keep_largest_component=cfg.pipeline.surfaces.preprocess.keep_largest_component)
        StructureContext = self.module("orchestrator").StructureContext
        if self.package == METHOD_PACKAGE["thalweg"]:
            margin_ = self.module("margin")
            t0 = time.time()
            margin, geometry, ref = margin_.union_margin(store, classes)
            timings = {"margin": time.time() - t0}
            if paths.debug:
                margin_.write_margin_image(paths.margin, margin, geometry)
            return StructureContext(**common, margin=margin, geometry=geometry, ref=ref, timings=timings)
        from vescan_field.field import margin_from_store
        margin, geom, qa, timings = margin_from_store(store, [store.class_index[c] for c in classes], ctGeom,
                                                      mask.shape, mask=mask, pad_voxels=cfg.pipeline.field.pad_voxels)
        if paths.debug:
            self.module("io").write_margin_image(paths.margin, margin, geom)
        return StructureContext(**common, margin=margin, geom=geom, qa=qa, timings=dict(timings))

    def _summary(self, cfg, ctx, store):
        """Start of run_summary.json, as the package's orchestrator writes it."""
        summary = dict(pipeline=self.package.removeprefix("vescan_"), patient=ctx.patient, structure=ctx.structure,
                       store_classes=ctx.classes, store=str(store.path))
        if self.package == METHOD_PACKAGE["thalweg"]:
            import thalweg
            return dict(summary, thalweg_version=thalweg.__version__, margin_grid=list(ctx.margin.shape))
        return dict(summary, field=dataclasses.asdict(cfg.pipeline.field), qa_mask=ctx.qa)

    def cancel(self):
        self._cancelRequested = True

    # 3D display

    def _showStageOutput(self, stage, ctx):
        kind, attr = STAGE_OUTPUT[stage]
        path = getattr(ctx.paths, attr)
        if path.exists():
            self._showModel(kind, ctx.structure, path)

    def _showModel(self, kind, structure, path):
        """Load the file into the model of (kind, structure), creating it the first time."""
        node = self._models.get((kind, structure))
        if node and slicer.mrmlScene.IsNodePresent(node):
            node.GetStorageNode().SetFileName(str(path))
            node.GetStorageNode().ReadData(node)
        else:
            node = self._models[(kind, structure)] = slicer.util.loadModel(str(path))
            node.SetAttribute(MODEL_ATTRIBUTE, f"{kind}/{structure}")
            node.SetName(f"VeScan {structure} {kind}" if structure else f"VeScan {kind}")
            color, opacity = MODEL_STYLE[kind]
            display = node.GetDisplayNode()
            display.SetColor(STRUCTURE_COLOR.get(structure, color) if color is None else color)
            display.SetOpacity(opacity)
            display.SetLineWidth(3)
            self._addToFolder(node)
        if kind in ("surface", "centerline"):
            self._colorModel(node, *self.colorBy)
        elif kind == "lobes":
            self._colorModel(node, self.lobesColorBy[0], LOBES_TABLE, self.lobesColorBy[1])
        elif kind == "fissures":
            # FissureContact, not Fissure: it splits the right oblique fissure into upper and lower part
            self._colorModel(node, "FissureContact", LOBES_TABLE, FISSURE_VALUES)
        self.updateLegends()
        slicer.app.processEvents()
        return node

    def colorTable(self, name, values=None):
        """VeScan colour table node, loaded once from Resources/ColorTables; with values, a copy with
        only those entries (same colours), so that its legend lists only them."""
        node = slicer.mrmlScene.GetFirstNode(name, "vtkMRMLColorTableNode")
        node = node or slicer.util.loadColorTable(str(COLOR_TABLES_DIR / f"{name}.csv"))
        if values is None:
            return node
        subsetName = f"{name} {'-'.join(map(str, values))}"
        subset = slicer.mrmlScene.GetFirstNode(subsetName, "vtkMRMLColorTableNode")
        if not subset:
            subset = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", subsetName)
            subset.SetTypeToUser()
            subset.SetNumberOfColors(node.GetNumberOfColors())
            subset.GetLookupTable().SetRange(node.GetLookupTable().GetRange())
            rgba = [0.0] * 4
            for value in values:
                node.GetColor(value, rgba)
                subset.SetColor(value, node.GetColorName(value), *rgba)
        return subset

    def _colorModel(self, node, arrayName, tableName, values=None):
        """Colour a model by a point or cell array with a VeScan table (None: Rainbow on the data
        range); a model without the array keeps its plain colour."""
        polyData = node.GetPolyData()
        if polyData.GetPointData().GetArray(arrayName) is not None:
            association = vtk.vtkAssignAttribute.POINT_DATA
        elif polyData.GetCellData().GetArray(arrayName) is not None:
            association = vtk.vtkAssignAttribute.CELL_DATA
        else:
            node.GetDisplayNode().SetScalarVisibility(False)
            return
        display = node.GetDisplayNode()
        display.SetActiveScalar(arrayName, association)
        if tableName:
            display.SetAndObserveColorNodeID(self.colorTable(tableName, values).GetID())
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseColorNodeScalarRange)
        else:
            display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange)
        display.SetScalarVisibility(True)

    def _showMarkups(self, kind, structure, path):
        old = self._models.get((kind, structure))
        if old and slicer.mrmlScene.IsNodePresent(old):
            slicer.mrmlScene.RemoveNode(old)
        node = self._models[(kind, structure)] = slicer.util.loadMarkups(str(path))
        node.SetAttribute(MODEL_ATTRIBUTE, f"{kind}/{structure}")
        node.SetName(f"VeScan {structure} {kind}")
        node.GetDisplayNode().SetPointLabelsVisibility(False)
        self._addToFolder(node)

    def _addToFolder(self, node):
        sh = slicer.mrmlScene.GetSubjectHierarchyNode()
        folder = sh.GetItemChildWithName(sh.GetSceneItemID(), "VeScan") or sh.CreateFolderItem(sh.GetSceneItemID(), "VeScan")
        sh.SetItemParent(sh.GetItemByDataNode(node), folder)

    # Cut tree

    def hasCenterline(self, structure) -> bool:
        node = self._models.get(("centerline", structure))
        return bool(node) and slicer.mrmlScene.IsNodePresent(node)

    def canCut(self, structure) -> bool:
        """Apply cut reads the run's files (vescan's clip stage)."""
        return self.hasCenterline(structure) and structure in self.contexts

    def getGenerationRange(self, structure):
        generation = slicer.util.arrayFromModelCellData(self._models[("centerline", structure)], "Generation")
        return int(generation.min()), int(generation.max())

    def runLobes(self) -> list:
        """Lobes of the run (keys of the contexts' lobes), in LOBES order."""
        lobes = {lobe for ctx in self.contexts.values() for lobe in ctx.lobes}
        return [lobe for lobe in LOBES if lobe in lobes]

    def canCutByLobe(self, structure) -> bool:
        """The cut limited to some lobes needs the Reaches_<LOBE> arrays of anatomical_segments.vtk (stage anatomy)."""
        return structure in self.contexts and self.contexts[structure].paths.anatomical_segments.exists()

    def generationLobes(self, lobes):
        """pipeline.clip.generation_lobes for the checked lobes: None when they are all the lobes of the run."""
        lobes = [lobe for lobe in self.runLobes() if lobe in lobes]
        return None if set(lobes) == set(self.runLobes()) else lobes

    def _previewGeneration(self, node, lobes):
        """Generation of the centerline cells, -1 for the groups kept at any generation: the ones that reach a
        lobe outside `lobes` (as vescan_field.stages.clip.cut.groups_outside)."""
        from vtk.util.numpy_support import numpy_to_vtk
        polyData = node.GetPolyData()
        generation = slicer.util.arrayFromModelCellData(node, "Generation").astype(np.int32)
        if lobes is not None:
            group = slicer.util.arrayFromModelCellData(node, "GroupId").astype(int)
            exempt = set()
            for lobe in set(self.runLobes()) - set(lobes):
                if polyData.GetCellData().GetArray(f"Reaches_{lobe}") is not None:
                    exempt |= set(group[slicer.util.arrayFromModelCellData(node, f"Reaches_{lobe}") > 0].tolist())
            generation[np.isin(group, list(exempt))] = -1
        array = numpy_to_vtk(generation, deep=True)
        array.SetName(PREVIEW_ARRAY)
        polyData.GetCellData().RemoveArray(PREVIEW_ARRAY)
        polyData.GetCellData().AddArray(array)
        polyData.Modified()

    def previewCut(self, structure, generations, lobes=None):
        """Hide the centerline cells with Generation > generations (None: show all), only in `lobes` (None: all,
        see generationLobes). The display threshold works only with scalars shown, so the preview colours
        the centerline by generation."""
        node = self._models[("centerline", structure)]
        display = node.GetDisplayNode()
        if generations is None:
            display.SetThresholdEnabled(False)
            return
        if not self.canCutByLobe(structure):
            lobes = None
        self._previewGeneration(node, lobes)
        display.SetActiveScalar(PREVIEW_ARRAY, vtk.vtkAssignAttribute.CELL_DATA)
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
        display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange)
        display.SetScalarVisibility(True)
        display.SetThresholdRange(-1, generations)
        display.SetThresholdEnabled(True)
        self.updateLegends()

    def applyCut(self, structure, generations, cap=True, lobes=None):
        """vescan's clip stage; lobes: pipeline.clip.generation_lobes (None: all the lobes, see generationLobes)."""
        ClipConfig, clip = self.module("config").ClipConfig, self.module("stages.clip")
        ctx = self.contexts[structure]
        if not self.canCutByLobe(structure):
            lobes = None
        clip.run(ctx, ClipConfig(max_generations=generations, cap=cap, generation_lobes=lobes))
        self._showModel("surface", structure, ctx.paths.clipped_surface)
        self._showMarkups("cutPoints", structure, ctx.paths.cut_points)

    # Results

    def _readStatistics(self):
        """All the rows of vescan's statistics.csv of every structure (for the export)."""
        self.statisticsRows = []
        for ctx in self.contexts.values():
            if ctx.paths.statistics_csv.exists():
                with open(ctx.paths.statistics_csv, newline="") as f:
                    self.statisticsRows += [row for row in csv.DictReader(f) if row.get("crossing")]

    def hasResults(self, structure=None) -> bool:
        """Any model of the module in the scene (of one structure, or of any)."""
        return any(slicer.mrmlScene.IsNodePresent(node) for (_kind, s), node in self._models.items()
                   if structure in (None, s))

    def restoreFromScene(self, outputDir):
        """Find the module's nodes in a loaded scene (by their attributes) and, if the run's files are still in
        outputDir, what Apply cut and the export need (vescan's clip stage reads only the context's paths)."""
        self._inputs, self._models, self.contexts, self._sceneVisibility = {}, {}, {}, {}
        self.lobesNode = self.crossingsTableNode = None
        for i in range(slicer.mrmlScene.GetNumberOfNodes()):
            node = slicer.mrmlScene.GetNthNode(i)
            if node.GetAttribute(INPUT_ATTRIBUTE):
                key = json.loads(node.GetAttribute(INPUT_KEY_ATTRIBUTE))
                self._inputs[node.GetAttribute(INPUT_ATTRIBUTE)] = (tuple(key) if isinstance(key, list) else key, node)
            elif node.GetAttribute(MODEL_ATTRIBUTE):
                kind, structure = node.GetAttribute(MODEL_ATTRIBUTE).split("/", 1)
                self._models[(kind, structure)] = node
            elif node.GetAttribute(TABLE_ATTRIBUTE) == "crossings":
                self.crossingsTableNode = node
        self.lobesNode = self._inputs.get("lobes", (None, None))[1]
        self.package = METHOD_PACKAGE.get(self.getParameterNode().method, METHOD_PACKAGE["field"])  # of the saved run
        if outputDir and importlib.util.find_spec(self.package):
            pathsmod = self.module("paths")
            PipelinePaths, final_export_dir = pathsmod.PipelinePaths, pathsmod.final_export_dir
            configPath = pathlib.Path(outputDir) / "vescan_config.json"
            enabled = json.loads(configPath.read_text())["pipeline"]["lobes"].get("enabled") if configPath.exists() else None
            lobes = dict.fromkeys(enabled or LOBES)  # the clip stage uses only the lobe names
            for structure in STRUCTURES:
                paths = PipelinePaths(final_export_dir(outputDir, structure))
                if self.hasCenterline(structure) and paths.branch_tree.exists():
                    self.contexts[structure] = types.SimpleNamespace(structure=structure, paths=paths, lobes=lobes,
                                                                     timings={})
        self._readStatistics()

    def loadResults(self):
        """Crossing vessels of every structure (vescan's statistics.csv) in one table node."""
        self._readStatistics()
        if not self.crossingsTableNode or not slicer.mrmlScene.IsNodePresent(self.crossingsTableNode):
            self.crossingsTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "VeScan crossings")
            self.crossingsTableNode.SetAttribute(TABLE_ATTRIBUTE, "crossings")
            self.crossingsTableNode.SetUseColumnTitleAsColumnHeader(True)
            self.crossingsTableNode.SetLocked(True)
            for name in CROSSING_COLUMNS:
                self.crossingsTableNode.AddColumn().SetName(name)
        table = self.crossingsTableNode
        table.GetTable().SetNumberOfRows(0)
        for row in self.statisticsRows:
            r = table.AddEmptyRow()
            values = [row["crossing"], row["structure"], row["crossing_type"], f"{row['from_lobe']} > {row['to_lobe']}",
                      row["length_mm"], row["angle_median_deg"]]
            for c, value in enumerate(values):
                table.SetCellText(r, c, value)
        table.Modified()
        return table

    def setVisibility(self, kind, visible, structure=None):
        """Show or hide the models of a kind, of one structure (None: all)."""
        for (k, s), node in self._models.items():
            if k in VISIBILITY_KINDS[kind] and structure in (None, s) and slicer.mrmlScene.IsNodePresent(node):
                node.SetDisplayVisibility(visible)
        self.updateLegends()

    def setColorBy(self, arrayName, tableName):
        """Colour the surfaces and centerlines (now and the ones shown later) by this array."""
        self.colorBy = (arrayName, tableName)
        for (kind, _structure), node in self._models.items():
            if kind in ("surface", "centerline") and slicer.mrmlScene.IsNodePresent(node):
                self._colorModel(node, arrayName, tableName)
        self.updateLegends()

    def setLobesColorBy(self, arrayName, values):
        """Colour the lobe mesh (now and when shown later) by one of its labellings."""
        self.lobesColorBy = (arrayName, values)
        node = self._models.get(("lobes", ""))
        if node and slicer.mrmlScene.IsNodePresent(node):
            self._colorModel(node, arrayName, LOBES_TABLE, values)
        self.updateLegends()

    def updateLegends(self):
        """One colour legend per group in the 3D view, on the first shown model coloured by an array:
        vessels (Color by) on the right, lobes (Lobes color by) on the left."""
        colorsLogic = slicer.modules.colors.logic()
        # width of a legend with names: it sets the text size (the font size of the legend is not used)
        for kinds, x, namesWidth in ((("centerline", "surface"), 0.9, 0.25), (("lobes",), 0.02, 0.4)):
            nodes = sorted(((kinds.index(k), node) for (k, _structure), node in self._models.items()
                            if k in kinds and slicer.mrmlScene.IsNodePresent(node)), key=lambda kn: kn[0])
            shown = [node for _i, node in nodes if self.showLegends and node.GetDisplayVisibility()
                     and node.GetDisplayNode().GetScalarVisibility()]
            for _i, node in nodes:
                legend = colorsLogic.GetColorLegendDisplayNode(node)
                if legend and (not shown or node is not shown[0]):
                    legend.SetVisibility(False)
            if shown:
                legend = colorsLogic.GetColorLegendDisplayNode(shown[0]) or colorsLogic.AddDefaultColorLegendDisplayNode(shown[0])
                display = shown[0].GetDisplayNode()
                legend.SetTitleText(display.GetActiveScalarName())
                # VeScan tables: one entry per named class; Rainbow (Generation): continuous bar
                names = display.GetScalarRangeFlag() == display.UseColorNodeScalarRange
                legend.SetUseColorNamesForLabels(names)
                legend.SetLabelFormat(legend.GetDefaultTextLabelFormat() if names else legend.GetDefaultNumericLabelFormat())
                legend.SetPosition(x, 0.5)
                legend.SetSize(namesWidth if names else 0.08, 0.6)  # the text is scaled to the legend width
                legend.SetVisibility2D(False)
                legend.SetVisibility(True)

    def focusCrossing(self, structure, crossingId):
        """Centre the slice views and the 3D camera on a crossing vessel."""
        node = self._models.get(("centerline", structure))
        if not node:
            return
        ids = slicer.util.arrayFromModelPointData(node, "CrossingIdPoint")
        points = slicer.util.arrayFromModelPoints(node)[ids == crossingId]
        if not len(points):
            return
        center = points.mean(0)
        slicer.modules.markups.logic().JumpSlicesToLocation(*center, True)
        viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
        slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode).SetFocalPoint(*center)

    def exportStatistics(self, path):
        """All the columns of vescan's statistics.csv, every structure in one file."""
        if not self.statisticsRows:
            raise ValueError("No statistics to export")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.statisticsRows[0]))
            writer.writeheader()
            writer.writerows(self.statisticsRows)


#
# VeScanTest
#


class VeScanTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_Geometry()
        self.test_Config()
        slicer.mrmlScene.Clear()  # the tests change the module's parameter node: start the module from a clean one

    def test_Geometry(self):
        """A volume node gives the same Geometry as vescan reading its file with SimpleITK."""
        import tempfile
        from vescan_field.field import load_reference
        node = slicer.util.addVolumeFromArray(np.zeros((4, 5, 6), np.uint8))
        node.SetSpacing(0.7, 0.8, 2.5)
        node.SetOrigin(-10, 20, 30)
        node.SetIJKToRASDirections(-1, 0, 0, 0, 0, 1, 0, -1, 0)
        path = pathlib.Path(tempfile.mkdtemp()) / "volume.nii.gz"
        slicer.util.saveNode(node, str(path))
        geom, reference = VeScanLogic.geometry(node), load_reference(path)[0]
        for attr in ("spacing", "origin", "direction"):
            self.assertTrue(np.allclose(getattr(geom, attr), getattr(reference, attr)), attr)
        self.delayDisplay("Geometry test passed")

    def test_Config(self):
        """The bundled JSON with the UI keys is a valid vescan config."""
        import tempfile
        logic = VeScanLogic()
        params = logic.getParameterNode()
        params.storePath = pathlib.Path("store.duckn.zip")
        params.stageAnatomy = False
        outputDir = pathlib.Path(tempfile.mkdtemp())
        cfg = logic.buildConfig(params, outputDir)
        self.assertFalse(cfg.pipeline.stages.anatomy)
        self.assertEqual(cfg.batch.structures.airways.classes, ["lung_airways"])
        self.assertTrue((outputDir / "vescan_config.json").exists())
        # Thalweg: its own JSON (no pipeline.field), with the bifurcations from the UI
        params.method, params.bifurcations = "thalweg", "point"
        cfg = logic.buildConfig(params, outputDir)
        self.assertEqual(cfg.pipeline.centerlines.bifurcations, "point")
        self.assertEqual(cfg.batch.structures.airways.classes, ["lung_airways"])
        self.delayDisplay("Config test passed")
