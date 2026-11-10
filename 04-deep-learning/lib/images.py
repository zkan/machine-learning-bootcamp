"""Fashion-MNIST files, or scikit-learn digits when those files are absent.

The IDX files are not in git. `python -m images --download` fetches them from
the Zalando research repo. Tests call `load_dataset("digits")` and stay offline.
"""

import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

MODULE = Path(__file__).resolve().parents[1]
FASHION_DIR = MODULE / "data" / "fashion"
SEED = 42

# Official files in https://github.com/zalandoresearch/fashion-mnist (MIT).
FASHION_BASE = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
)
FASHION_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def fashion_ready(folder: Path = FASHION_DIR) -> bool:
    return all((folder / name).is_file() for name in FASHION_FILES)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
    if magic != 2051:
        raise ValueError(f"{path.name} is not an image IDX file")
    raw = gzip.open(path, "rb")
    raw.read(16)
    data = np.frombuffer(raw.read(), dtype=np.uint8)
    raw.close()
    return data.reshape(count, rows, cols).copy()


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if magic != 2049:
        raise ValueError(f"{path.name} is not a label IDX file")
    if len(data) != count:
        raise ValueError(f"{path.name} label count does not match the header")
    return data.astype(np.int64).copy()


def load_fashion(folder: Path = FASHION_DIR) -> dict:
    images = np.concatenate(
        [
            _read_idx_images(folder / "train-images-idx3-ubyte.gz"),
            _read_idx_images(folder / "t10k-images-idx3-ubyte.gz"),
        ]
    )
    labels = np.concatenate(
        [
            _read_idx_labels(folder / "train-labels-idx1-ubyte.gz"),
            _read_idx_labels(folder / "t10k-labels-idx1-ubyte.gz"),
        ]
    )
    return {
        "source": "fashion",
        "images": images,
        "labels": labels,
        "classes": FASHION_CLASSES,
        "image_size": int(images.shape[1]),
    }


def load_digit_images() -> dict:
    digits = load_digits()
    images = np.asarray(digits.images)
    # sklearn stores 0-16 ink levels. Scale to 0-255 so the CNN sees the same range as Fashion-MNIST.
    images = np.clip(np.rint(images * (255 / 16)), 0, 255).astype(np.uint8)
    return {
        "source": "digits",
        "images": images,
        "labels": np.asarray(digits.target, dtype=np.int64),
        "classes": tuple(str(i) for i in range(10)),
        "image_size": int(images.shape[1]),
    }


def load_dataset(source: str = "auto") -> dict:
    if source == "digits":
        return load_digit_images()
    if source == "fashion":
        if not fashion_ready():
            raise FileNotFoundError(
                f"Fashion-MNIST files are not in {FASHION_DIR}. Run python -m images --download."
            )
        return load_fashion()
    if source != "auto":
        raise ValueError("source must be auto, fashion, or digits")
    if fashion_ready():
        return load_fashion()
    return load_digit_images()


def split_dataset(bundle: dict, seed: int = SEED):
    """Train 60 percent, validation 20 percent, held-out 20 percent. Stratified."""
    images, labels = bundle["images"], bundle["labels"]
    rest_x, hold_x, rest_y, hold_y = train_test_split(
        images, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    train_x, val_x, train_y, val_y = train_test_split(
        rest_x, rest_y, test_size=0.25, random_state=seed, stratify=rest_y
    )
    return (train_x, train_y), (val_x, val_y), (hold_x, hold_y)


def take(images, labels, limit: int | None):
    if limit is None or len(images) <= limit:
        return images, labels
    return images[:limit], labels[:limit]


def download_fashion(folder: Path = FASHION_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in FASHION_FILES:
        dest = folder / name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"have {name}")
            continue
        url = FASHION_BASE + name
        print(f"get {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"wrote {dest} bytes {dest.stat().st_size}")


def main(argv: list[str]) -> int:
    if "--download" in argv:
        download_fashion()
        bundle = load_fashion()
        print(f"fashion images {len(bundle['images'])} size {bundle['image_size']}")
        return 0
    if "--check" in argv:
        if not fashion_ready():
            digits = load_digit_images()
            print(f"fashion files missing; digits fallback rows {len(digits['images'])}")
            return 0
        bundle = load_fashion()
        print(f"fashion files match rows {len(bundle['images'])}")
        return 0
    print("usage: python -m images --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
