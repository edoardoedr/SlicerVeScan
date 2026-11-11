"""Sync SlicerVeScan with a release (tag) of VeScan-field.

    python tools/sync_vescan.py v1.1.1

1. downloads the tag's archive from GitHub (the one the users install);
2. checks that its pyproject.toml has the tag's version;
3. writes the two requirement files read by VeScan.ensureVescanField, from the release's dependency tree:
   - VeScan/Resources/requirements.txt: vescan-field and every dependency pinned to a git URL or an archive
     (rankfield, thalweg, labelfield, ...), as GitHub archives, installed with --no-deps: no git is needed and
     pip never compares two URLs of the same package (thalweg asks rankfield from git, vescan-field as a zip);
   - VeScan/Resources/requirements-deps.txt: their PyPI dependencies (extras included), installed normally,
     without the packages that are Slicer's own (numpy, scipy, vtk, SimpleITK);
   the tree is walked through each archive's pyproject.toml;
4. copies the four ColorTable*_EN.csv into VeScan/Resources/ColorTables/;
5. compares configs/<method>/example.json with VeScan/Resources/vescan_<method>.json, for each method (field,
   thalweg), without changing them: the keys set by the UI are skipped, a new, removed or changed key may need
   a change in the module.

No commit, nothing installed. Standard library only (Python >= 3.11, for tomllib).
"""
import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "VeScan" / "Resources"
ARCHIVE_URL = "https://github.com/edoardoedr/VeScan-field/archive/refs/tags/{tag}.zip"
METHODS = ["field", "thalweg"]  # configs/<method>/example.json <-> VeScan/Resources/vescan_<method>.json
COLOR_TABLES = ["ColorTableCrossings_EN.csv", "ColorTableLobeLabel_EN.csv",
                "ColorTableLobesFissures_EN.csv", "ColorTableVessels_EN.csv"]
SLICER_PACKAGES = {"numpy", "scipy", "vtk", "simpleitk"}  # Slicer's own: never installed by the module
ARCHIVES_HEADER = """\
# Installed by VeScan.ensureVescanField with --no-deps (then requirements-deps.txt); written by
# tools/sync_vescan.py from vescan-field {version}'s dependency tree, do not edit by hand.
# GitHub archives instead of git+https: no git is needed on the user's machine, and pip never compares
# two URLs of the same package (thalweg asks rankfield from git, vescan-field as an archive).
"""
DEPS_HEADER = """\
# PyPI dependencies of requirements.txt (installed with Slicer's numpy pinned); written by
# tools/sync_vescan.py from vescan-field {version}'s dependency tree, do not edit by hand.
# numpy, scipy, vtk and SimpleITK are Slicer's own and are not listed.
"""
# keys of the config set by the module (VeScanLogic.buildConfig / applyCut) or only descriptive
UI_KEYS = re.compile(r"^(experiment_name|notes|batch\.(patients|surfaces|results)_root_dir|batch\.store"
                     r"|batch\.structures\.\w+\.enabled|pipeline\.stages\.\w+|pipeline\.clip\.\w+"
                     r"|pipeline\.lobes\.enabled|pipeline\.centerlines\.bifurcations)$")
# name[extras] @ url ; marker   or   name[extras] specifier ; marker
REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?\s*(?:@\s*(\S+)|([^;]*))\s*(?:;.*)?$")


def flatten(data, prefix=""):
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def archive_url(url):
    """GitHub archive of a direct URL: git+https://github.com/<o>/<r>.git@<ref> -> .../archive/<ref>.zip."""
    git = re.match(r"git\+(https://github\.com/[^@]+?)(?:\.git)?@(\S+)$", url)
    if not git:
        if url.endswith(".zip"):
            return url
        sys.exit(f"Unsupported dependency URL (no pinned ref, or not on GitHub): {url}")
    repo, ref = git.groups()
    kind = "" if re.fullmatch(r"[0-9a-f]{40}", ref) else "refs/tags/"
    return f"{repo}/archive/{kind}{ref}.zip"


def download(url, folder):
    """Extract the archive at url into a new subfolder of folder, return its root."""
    target = Path(tempfile.mkdtemp(dir=folder))
    with urllib.request.urlopen(url) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(target)
    return next(target.iterdir())


def dependency_tree(release, folder):
    """(archives, pypi) of the release's dependency tree: name -> archive URL, requirement -> set of specifiers."""
    archives, pypi = {}, {}

    def walk(root, extras=()):
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        requirements = list(project.get("dependencies", []))
        for extra in extras:
            requirements += project.get("optional-dependencies", {}).get(extra, [])
        for requirement in requirements:
            match = REQUIREMENT.match(requirement)
            if not match:
                sys.exit(f"Cannot parse the requirement {requirement!r} of {project['name']}")
            name, subExtras, url, specifier = match.groups()
            key = normalize(name)
            subExtras = [e.strip() for e in (subExtras or "").split(",") if e.strip()]
            if url:
                url = archive_url(url)
                if key in archives and archives[key] != url:
                    sys.exit(f"{name}: two different versions in the tree: {archives[key]} and {url}")
                if key not in archives:
                    archives[key] = url
                    walk(download(url, folder), subExtras)
            elif key not in SLICER_PACKAGES:
                pypi.setdefault(key, set()).update(s.strip() for s in (specifier or "").split(",") if s.strip())

    walk(release)
    return archives, pypi


def write(path, text):
    changed = not path.exists() or path.read_text() != text
    path.write_text(text)
    print(f"{path.name}: {'updated' if changed else 'unchanged'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="VeScan-field tag, e.g. v1.1.1")
    tag = parser.parse_args().tag
    version = tag.removeprefix("v")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            release = download(ARCHIVE_URL.format(tag=tag), tmp)
        except Exception as error:
            sys.exit(f"Cannot download {ARCHIVE_URL.format(tag=tag)}: {error}")

        declared = re.search(r'^version = "([^"]+)"', (release / "pyproject.toml").read_text(), re.M).group(1)
        if declared != version:
            sys.exit(f"{tag}: pyproject.toml has version {declared}, not {version}. Fix the release first.")

        archives, pypi = dependency_tree(release, tmp)
        # the dependencies, then vescan-field (ensureVescanField reads the required version from its line)
        lines = [f"{name} @ {url}" for name, url in sorted(archives.items())]
        lines.append(f"vescan-field @ {ARCHIVE_URL.format(tag=tag)}")
        write(RESOURCES / "requirements.txt", ARCHIVES_HEADER.format(version=version) + "\n".join(lines) + "\n")
        deps = [name + ",".join(sorted(specs)) for name, specs in sorted(pypi.items())]
        write(RESOURCES / "requirements-deps.txt", DEPS_HEADER.format(version=version) + "\n".join(deps) + "\n")
        print(f"  from git/archives (--no-deps): {', '.join(sorted(archives))}, vescan-field")
        print(f"  from PyPI: {', '.join(deps)}")

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
