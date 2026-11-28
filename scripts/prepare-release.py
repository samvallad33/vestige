#!/usr/bin/env python3
"""Prepare local release artifacts; never tag, upload, release, or publish.

Archive checks also apply before npm staging. A checksum proves byte identity,
not authorship: keep the CI run and reviewed source commit with the artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "aarch64-apple-darwin": "darwin-arm64",
    "x86_64-apple-darwin": "darwin-x64",
    "x86_64-unknown-linux-gnu": "linux-x64",
    "aarch64-unknown-linux-gnu": "linux-arm64",
    "x86_64-pc-windows-msvc": "win32-x64",
    "aarch64-linux-android": None,
}
BINARIES = ["vestige-mcp", "vestige", "vestige-restore", "vestige-upgrade"]
MAX_BINARY_BYTES = 256 * 1024 * 1024
JSON_MANIFESTS = [
    "package.json", "apps/dashboard/package.json", "packages/vestige-init/package.json",
    "packages/vestige-mcp-npm/package.json", "packages/vestige-mcpb/manifest.json",
    "server.json", "plugins/vestige-strata/package.json",
    "plugins/vestige-strata/openclaw.plugin.json",
    # Legacy standalone operator-lite npm/openclaw packages & lhm.plugin.json removed in v7.1 consolidation.
] + [f"packages/mcp-{platform}/package.json" for platform in TARGETS.values() if platform]
CARGO_MANIFESTS = ["Cargo.toml"] + [
    f"crates/{name}/Cargo.toml" for name in ("vestige-core", "vestige-mcp", "vestige-upgrade", "strata", "strata-gate",
                                   "strata-kernel", "strata-migrate", "strata-store", "strata-verify")
]


def digest(file):
    value = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expected_files(target):
    names = ["vestige-mcp", "vestige-upgrade"] if target.endswith("android") else BINARIES
    return [f"{name}.exe" if target.endswith("msvc") else name for name in names]


def archive_name(target):
    suffix = "zip" if target.endswith("msvc") else "tar.gz"
    return f"vestige-mcp-{target}.{suffix}"


def check_version(version, root=ROOT):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("version must have the form X.Y.Z")
    for name in JSON_MANIFESTS:
        package = json.loads((root / name).read_text())
        if package["version"] != version:
            raise ValueError(f"{name}: {package['version']} does not match {version}")
        for dependency, pinned in package.get("optionalDependencies", {}).items():
            if dependency.startswith("@vestige/mcp-") and pinned != version:
                raise ValueError(f"{name}: {dependency} must pin {version}")
        for entry in package.get("packages", []):
            if entry["version"] != version:
                raise ValueError(f"{name}: registry entry must use {version}")
    for name in CARGO_MANIFESTS:
        text = (root / name).read_text()
        found = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
        if not found or found[1] != version:
            raise ValueError(f"{name}: product version does not match {version}")
    core_dependency = re.search(
        r'vestige-core = \{ version = "([^"]+)"',
        (root / "crates/vestige-mcp/Cargo.toml").read_text(),
    )
    if not core_dependency or core_dependency[1] != version:
        raise ValueError("vestige-mcp must depend on the matching vestige-core version")
    print(f"Product manifests and npm binary pins match {version}")


def package_archive(target, build_root, output):
    files = expected_files(target)
    source = build_root / target / "release"
    for name in files:
        file = source / name
        if not file.is_file() or file.is_symlink() or not file.stat().st_size:
            raise ValueError(f"missing, empty, or symlinked build output: {file}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / archive_name(target)
    if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
        raise ValueError(f"refusing to overwrite staged archive {archive}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in files:
                bundle.write(source / name, name)
    else:
        with tarfile.open(archive, "w:gz") as bundle:
            for name in files:
                bundle.add(source / name, arcname=name, recursive=False)
    checksum = digest(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"target": target, "archive": str(archive), "sha256": checksum}))


def inspect_archive(target, assets, destination=None):
    archive = assets / archive_name(target)
    checksum_file = archive.with_name(archive.name + ".sha256")
    if archive.is_symlink() or checksum_file.is_symlink():
        raise ValueError(f"symlinked release asset: {archive}")
    expected = checksum_file.read_text().strip().split()
    actual = digest(archive)
    if len(expected) != 2 or expected != [actual, archive.name]:
        raise ValueError(f"checksum mismatch or malformed checksum for {archive.name}")
    required = set(expected_files(target))
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) != len(required) or {item.filename for item in members} != required:
                raise ValueError(f"unexpected archive entries in {archive.name}")
            for item in members:
                if (item.is_dir() or ((item.external_attr >> 16) & 0o170000) == 0o120000
                        or not 0 < item.file_size <= MAX_BINARY_BYTES):
                    raise ValueError(f"nonregular, empty, or oversized archive member: {item.filename}")
                if destination:
                    (destination / item.filename).write_bytes(bundle.read(item))
    else:
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            if len(members) != len(required) or {item.name for item in members} != required:
                raise ValueError(f"unexpected archive entries in {archive.name}")
            for item in members:
                if not item.isfile() or not 0 < item.size <= MAX_BINARY_BYTES:
                    raise ValueError(f"nonregular, empty, or oversized archive member: {item.name}")
                if destination:
                    with bundle.extractfile(item) as stream:
                        (destination / item.name).write_bytes(stream.read())
    return {"target": target, "archive": archive.name, "sha256": actual, "bytes": archive.stat().st_size}


def verify_assets(assets):
    records = [inspect_archive(target, assets) for target in TARGETS]
    print(json.dumps({"archives": records}, indent=2))


def stage_npm(version, assets, output, root=ROOT):
    check_version(version, root)
    # Validate the complete six-target set before writing any package directory.
    for target in TARGETS:
        inspect_archive(target, assets)
    if output.exists():
        raise ValueError(f"staging directory already exists: {output}")
    output.mkdir(parents=True)
    for target, platform in TARGETS.items():
        if not platform:
            continue
        dest = output / f"mcp-{platform}"
        (dest / "bin").mkdir(parents=True)
        shutil.copy2(root / f"packages/mcp-{platform}/package.json", dest / "package.json")
        inspect_archive(target, assets, dest / "bin")
        for binary in (dest / "bin").iterdir():
            binary.chmod(0o755)
    for source_name in ("vestige-mcp-npm", "vestige-init"):
        dest = output / source_name
        source = root / "packages" / source_name
        for entry in source.rglob("*"):
            if "node_modules" not in entry.parts and entry.is_symlink():
                raise ValueError(f"symlink in npm package source: {entry}")
        shutil.copytree(source, dest,
                        ignore=shutil.ignore_patterns("node_modules", "*.tgz", "__pycache__"))
        manifest_path = dest / "package.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["optionalDependencies"] = {
            f"@vestige/mcp-{platform}": version for platform in TARGETS.values() if platform
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip())
    (output / "preparation.json").write_text(json.dumps({
        "version": version, "sourceCommit": source_commit, "sourceTreeDirty": dirty,
        "published": False, "note": "Local staging only; inspect source dirty state before shipping.",
        "archives": [inspect_archive(target, assets) for target in TARGETS],
    }, indent=2) + "\n")
    print(f"Staged five platform packages, vestige-mcp-server and @vestige/init in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("check-version")
    check.add_argument("--version", required=True)
    archive = subcommands.add_parser("archive")
    archive.add_argument("--target", required=True, choices=TARGETS)
    archive.add_argument("--build-root", type=Path, default=ROOT / "target")
    archive.add_argument("--output", type=Path, required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--assets", type=Path, required=True)
    npm = subcommands.add_parser("npm")
    npm.add_argument("--version", required=True)
    npm.add_argument("--assets", type=Path, required=True)
    npm.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check-version":
            check_version(args.version)
        elif args.command == "archive":
            package_archive(args.target, args.build_root, args.output)
        elif args.command == "verify":
            verify_assets(args.assets)
        else:
            stage_npm(args.version, args.assets, args.output)
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Release preparation failed: {error}\n")


if __name__ == "__main__":
    main()
