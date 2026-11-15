"""Pack and install the FIRMS Parquet artifacts as GitHub Release assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "release-manifest.json"
REPOSITORY = "Nasa-Planetarium/Planetarium-AI"
TAG = "firms-1km-backend-2026-11-14"
PART_LIMIT = 1792 * 1024**2
SCOPES = {"archive": "dataset", "daily": "predictions/daily", "annual": "predictions/annual"}


class TarSink:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, block):
        count = self.stream.write(block)
        self.digest.update(block[:count])
        return count


def pack(source, output, manifest, groups=SCOPES):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.tar")):
        raise FileExistsError(f"Use an empty output folder: {output}")
    entries = []
    for group, scope in SCOPES.items():
        if group not in groups:
            continue
        partition = source / scope
        if not (partition / "metadata.json").is_file():
            raise FileNotFoundError(partition / "metadata.json")
        paths = sorted(path for path in partition.rglob("*") if path.is_file() and path.suffix in (".parquet", ".csv", ".json"))
        batches = [[]]
        total = 0
        for file in paths:
            size = file.stat().st_size
            tar_size = 512 + ((size + 511) // 512) * 512
            if tar_size + 10240 >= PART_LIMIT:
                raise ValueError(f"Single file is too large for a release asset: {file}")
            if total + tar_size + 10240 >= PART_LIMIT:
                batches.append([])
                total = 0
            batches[-1].append((file, size))
            total += tar_size
        for index, batch in enumerate(batches, start=1):
            prefix = "firms-1km" if group == "archive" else "firms-1km-timesfm3"
            name = f"{prefix}-{group}-{index:02d}.tar"
            path = output / name
            partial = path.with_suffix(".part")
            with partial.open("wb") as stream:
                sink = TarSink(stream)
                with tarfile.open(fileobj=sink, mode="w|", format=tarfile.USTAR_FORMAT, bufsize=2**20) as archive:
                    for file, size in batch:
                        item = tarfile.TarInfo(file.relative_to(source).as_posix())
                        item.size = size
                        item.mode = 0o644
                        with file.open("rb") as data:
                            archive.addfile(item, data)
            partial.replace(path)
            entries.append({"name": name, "group": group, "bytes": path.stat().st_size,
                            "sha256": sink.digest.hexdigest(), "files": len(batch)})
            print(name, path.stat().st_size, flush=True)
    data = {"repository": REPOSITORY, "release_tag": TAG, "assets": entries}
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def install(destination, manifest, groups, local=None):
    specs = json.loads(manifest.read_text(encoding="utf-8"))
    entries = [entry for entry in specs["assets"] if entry["group"] in groups]
    if not entries:
        raise ValueError("No matching assets in the release manifest")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="firms-assets-") as temporary:
        for entry in entries:
            key = entry["name"]
            if Path(key).name != key or not key.endswith(".tar"):
                raise ValueError("Invalid asset name")
            if local is None:
                url = f"https://github.com/{REPOSITORY}/releases/download/{specs['release_tag']}/{key}"
                path = Path(temporary) / key
                with urlopen(url, timeout=90) as src, path.open("wb") as out:
                    shutil.copyfileobj(src, out, length=2**20)
            else:
                path = local / key
            with path.open("rb") as file:
                signature = hashlib.file_digest(file, "sha256").hexdigest()
            if path.stat().st_size != entry["bytes"] or signature != entry["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch for {key}")
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    relative = PurePosixPath(member.name)
                    if (not member.isfile() or relative.is_absolute() or ".." in relative.parts or
                            not member.name.startswith(SCOPES[entry["group"]] + "/")):
                        raise ValueError(f"Unexpected archive member for {key}: {member.name}")
                    archive.extract(member, destination, filter="data")
            print("Installed", key, flush=True)
            if local is None:
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("pack", help="Prepare the local release archives and manifest")
    create.add_argument("--source", type=Path, default=ROOT)
    create.add_argument("--output", type=Path, default=ROOT / ".release-assets")
    create.add_argument("--manifest", type=Path, default=MANIFEST)
    create.add_argument("--group", choices=SCOPES, action="append")
    retrieve = sub.add_parser("install", help="Install and verify release archives for this backend")
    retrieve.add_argument("--destination", type=Path, default=ROOT)
    retrieve.add_argument("--manifest", type=Path, default=MANIFEST)
    retrieve.add_argument("--group", choices=SCOPES, action="append")
    retrieve.add_argument("--local", type=Path, help="Read the archive files from a local directory")
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.source, args.output, args.manifest, args.group or SCOPES)
    else:
        install(args.destination, args.manifest, args.group or SCOPES, args.local)


if __name__ == "__main__":
    main()
