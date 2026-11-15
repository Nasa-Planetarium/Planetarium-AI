"""Install the harmonized FIRMS archive from the GitHub Release."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from .api import DATA_ROOT, ROOT

MANIFEST = ROOT / "release-manifest.json"


def install(destination, *, local=None, manifest=MANIFEST):
    destination = Path(destination).resolve()
    dataset = destination / "dataset"
    if dataset.exists():
        raise FileExistsError(f"A dataset is already installed at {dataset}")
    info = json.loads(Path(manifest).read_text(encoding="utf-8"))
    location = f"https://github.com/{info['repository']}/releases/download/{info['release_tag']}"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".firms-install-", dir=destination) as temporary:
        staging = Path(temporary)
        for entry in info["assets"]:
            name = entry["name"]
            if Path(name).name != name or not name.endswith(".tar"):
                raise ValueError("Invalid asset name in release manifest")
            path = Path(local) / name if local is not None else staging / name
            if local is None:
                with urlopen(f"{location}/{name}", timeout=90) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=2**20)
            with path.open("rb") as file:
                signature = hashlib.file_digest(file, "sha256").hexdigest()
            if path.stat().st_size != entry["bytes"] or signature != entry["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch for {name}")
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    relative = PurePosixPath(member.name)
                    if (not member.isfile() or relative.is_absolute() or ".." in relative.parts or
                            not member.name.startswith("dataset/")):
                        raise ValueError(f"Unexpected archive member in {name}: {member.name}")
                    archive.extract(member, staging, filter="data")
            if local is None:
                path.unlink()
            print("Verified", name, flush=True)
        if not (staging / "dataset" / "metadata.json").is_file():
            raise ValueError("The release did not contain dataset/metadata.json")
        (staging / "dataset").rename(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install"])
    parser.add_argument("--destination", type=Path, default=DATA_ROOT)
    parser.add_argument("--local", type=Path, help="Directory of already downloaded tar files")
    args = parser.parse_args()
    install(args.destination, local=args.local)


if __name__ == "__main__":
    main()
