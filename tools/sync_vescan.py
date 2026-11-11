"""Sync SlicerVeScan with a release (tag) of VeScan-field.

    python tools/sync_vescan.py v0.1.2

1. downloads the tag's archive from GitHub (the one the users install);
2. checks that its pyproject.toml has the tag's version;
3. points VeScan/Resources/requirements.txt to the tag (ensureVescanField reads the required version there);
4. copies the four ColorTable*_EN.csv into VeScan/Resources/ColorTables/;
5. compares configs/<method>/example.json with VeScan/Resources/vescan_<method>.json, for each method (field,
   thalweg), without changing them: the keys set by the UI are skipped, a new, removed or changed key may need
   a change in the module.

No commit, nothing installed. Standard library only.
"""
import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "VeScan" / "Resources"
ARCHIVE_URL = "https://github.com/edoardoedr/VeScan-field/archive/refs/tags/{tag}.zip"
METHODS = ["field", "thalweg"]  # configs/<method>/example.json <-> VeScan/Resources/vescan_<method>.json
COLOR_TABLES = ["ColorTableCrossings_EN.csv", "ColorTableLobeLabel_EN.csv",
                "ColorTableLobesFissures_EN.csv", "ColorTableVessels_EN.csv"]
# keys of the config set by the module (VeScanLogic.buildConfig / applyCut) or only descriptive
UI_KEYS = re.compile(r"^(experiment_name|notes|batch\.(patients|surfaces|results)_root_dir|batch\.store"
                     r"|batch\.structures\.\w+\.enabled|pipeline\.stages\.\w+|pipeline\.clip\.\w+"
                     r"|pipeline\.lobes\.enabled|pipeline\.centerlines\.bifurcations)$")


def flatten(data, prefix=""):
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def download(tag, folder):
    """Extract the tag's archive into folder, return its root."""
    with urllib.request.urlopen(ARCHIVE_URL.format(tag=tag)) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(folder)
    return next(Path(folder).iterdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="VeScan-field tag, e.g. v0.1.2")
    tag = parser.parse_args().tag
    version = tag.removeprefix("v")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            release = download(tag, tmp)
        except Exception as error:
            sys.exit(f"Cannot download {ARCHIVE_URL.format(tag=tag)}: {error}")

        declared = re.search(r'^version = "([^"]+)"', (release / "pyproject.toml").read_text(), re.M).group(1)
        if declared != version:
            sys.exit(f"{tag}: pyproject.toml has version {declared}, not {version}. Fix the release first.")

        requirements = RESOURCES / "requirements.txt"
        text = requirements.read_text()
        requirements.write_text(re.sub(r"refs/tags/[^/\s]+\.zip", f"refs/tags/{tag}.zip", text))
        print(f"requirements.txt: vescan-field {version}" + ("" if requirements.read_text() != text else " (unchanged)"))

        for name in COLOR_TABLES:
            source, target = release / name, RESOURCES / "ColorTables" / name
            changed = not target.exists() or source.read_bytes() != target.read_bytes()
            shutil.copyfile(source, target)
            print(f"{name}: {'updated' if changed else 'unchanged'}")

        examples = {method: json.loads((release / "configs" / method / "example.json").read_text())
                    for method in METHODS}
    for method, example in examples.items():
        compare(example, RESOURCES / f"vescan_{method}.json")


def compare(example, bundledPath):
    """Print the differences between a release's example.json and a bundled config (keys set by the UI skipped)."""
    example = flatten(example)
    bundled = flatten(json.loads(bundledPath.read_text()))
    keys = sorted(key for key in example.keys() | bundled.keys() if not UI_KEYS.match(key))
    differences = [f"  new key     {key} = {example[key]!r}" if key not in bundled else
                   f"  removed key {key}" if key not in example else
                   f"  changed     {key}: {bundled[key]!r} -> {example[key]!r}"
                   for key in keys if example.get(key, None) != bundled.get(key, None) or (key in example) != (key in bundled)]
    if differences:
        print(f"{bundledPath.name} differs from example.json (not changed, check the module):")
        print("\n".join(differences))
    else:
        print(f"{bundledPath.name}: in sync with example.json")


if __name__ == "__main__":
    main()
