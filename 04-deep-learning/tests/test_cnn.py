from pathlib import Path

import cv2

from cnn import classify_upload, predict, save_cnn, train_cnn
from cup import class_parts, instructor_labels, label_frame, prediction_frame, score_files, write_frame
from images import fashion_ready


def _fit(source):
    bundle, train, val, hold_x = class_parts(source)
    result = train_cnn(
        train[0], train[1], val[0], val[1], bundle["image_size"], len(bundle["classes"])
    )
    hold_y = instructor_labels(source)["label"].to_numpy()
    return bundle, result, hold_x, hold_y


def test_digits_fallback_is_offline_and_repeatable():
    bundle, result, hold_x, hold_y = _fit("digits")
    bundle2, train2, val2, _hold = class_parts("digits")
    again = train_cnn(
        train2[0], train2[1], val2[0], val2[1], bundle2["image_size"], len(bundle2["classes"])
    )
    assert bundle["source"] == "digits"
    assert round(result["validation_accuracy"], 3) == 0.406
    assert result["validation_accuracy"] == again["validation_accuracy"]
    assert result["seconds"] > 0
    held = float((predict(result["model"], hold_x) == hold_y).mean())
    assert round(held, 3) == 0.347


def test_fashion_subset_when_the_files_are_present():
    if not fashion_ready():
        return
    _bundle, result, hold_x, hold_y = _fit("fashion")
    assert result["validation_accuracy"] == 0.6975
    assert round(result["seconds"], 2) < 5
    held = float((predict(result["model"], hold_x) == hold_y).mean())
    assert round(held, 3) == 0.677


def test_upload_and_cup_score(tmp_path):
    bundle, result, hold_x, hold_y = _fit("digits")
    path = tmp_path / "cnn.pt"
    save_cnn(path, result["model"], bundle["image_size"], bundle["classes"], bundle["source"])
    gray = hold_x[0]
    ok, encoded = cv2.imencode(".png", gray)
    assert ok
    name, score = classify_upload(path, encoded.tobytes())
    assert name in bundle["classes"]
    assert 0 < score <= 1
    preds = tmp_path / "pred.csv"
    labels = tmp_path / "labels.csv"
    guessed = predict(result["model"], hold_x)
    write_frame(prediction_frame(guessed), preds)
    write_frame(label_frame(hold_y), labels)
    assert score_files(preds, labels) == float((guessed == hold_y).mean())
    assert not (Path("data") / "cup_labels.csv").exists()
