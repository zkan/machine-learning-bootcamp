"""A small grayscale CNN. No torchvision. Images stay NumPy until the training loop."""

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

SEED = 42
# Fashion-MNIST is 70,000 images. Class uses this prefix of each split so a CPU laptop finishes.
FASHION_TRAIN_LIMIT = 2000
FASHION_VAL_LIMIT = 400
FASHION_HOLD_LIMIT = 400
EPOCHS = 2
BATCH = 64


class SmallCNN(nn.Module):
    def __init__(self, image_size: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 16 * (image_size // 4) * (image_size // 4)
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, batch):
        return self.classifier(torch.flatten(self.features(batch), 1))


def _tensor(images: np.ndarray) -> torch.Tensor:
    array = images.astype(np.float32) / 255.0
    return torch.from_numpy(array[:, None, :, :])


def _batches(images, labels, batch_size):
    order = torch.randperm(len(images))
    for start in range(0, len(images), batch_size):
        pick = order[start : start + batch_size]
        yield images[pick], labels[pick]


def accuracy(model: nn.Module, images: np.ndarray, labels: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        scores = model(_tensor(images))
        pred = scores.argmax(dim=1).numpy()
    return float((pred == labels).mean())


def predict(model: nn.Module, images: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = model(_tensor(images))
    return scores.argmax(dim=1).numpy()


def train_cnn(
    train_x,
    train_y,
    val_x,
    val_y,
    image_size: int,
    n_classes: int,
    learning_rate: float = 0.001,
    epochs: int = EPOCHS,
    seed: int = SEED,
):
    torch.manual_seed(seed)
    model = SmallCNN(image_size, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    x_train = _tensor(train_x)
    y_train = torch.from_numpy(train_y.astype(np.int64))
    started = time.perf_counter()
    model.train()
    for _epoch in range(epochs):
        for xb, yb in _batches(x_train, y_train, BATCH):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    seconds = time.perf_counter() - started
    val_acc = accuracy(model, val_x, val_y)
    train_acc = accuracy(model, train_x, train_y)
    return {
        "model": model,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "seconds": seconds,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }


def confusion(model: nn.Module, images, labels, n_classes: int) -> np.ndarray:
    guessed = predict(model, images)
    return confusion_matrix(labels, guessed, labels=list(range(n_classes)))


def save_cnn(path: Path, model: nn.Module, image_size: int, classes, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": int(image_size),
            "classes": list(classes),
            "source": source,
        },
        path,
    )


def load_cnn(path: Path) -> tuple[nn.Module, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallCNN(int(blob["image_size"]), len(blob["classes"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def prepare_gray(gray: np.ndarray, image_size: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def classify_upload(path: Path, file_bytes: bytes) -> tuple[str, float]:
    """Return the class name and the score of that class for one image file."""
    import cv2

    model, blob = load_cnn(path)
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("OpenCV could not read that image.")
    gray = prepare_gray(decoded, int(blob["image_size"]))
    model.eval()
    with torch.no_grad():
        scores = model(_tensor(gray[None, ...]))
        probs = torch.softmax(scores, dim=1)[0]
        index = int(probs.argmax())
    return blob["classes"][index], float(probs[index])
