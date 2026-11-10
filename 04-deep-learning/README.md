# Day 3: Deep learning

Day 3 of the Machine Learning Bootcamp. You stay with Fresh and Good, the fictional Bangkok online grocery from Day 1. In the morning, fit a small neural net on churn and compare it with the random forest. After lunch, try OpenCV, train a small CNN on product images, and add an image tab to the form. At 15:30, log one run in MLflow and share the memo.

- 3.1 [Neural net on churn](00_mlp_churn.ipynb)
- 3.2 [Training behaviour](01_training.ipynb)
- 3.3 [Images with OpenCV](02_opencv.ipynb)
- 3.4 [A small CNN and the image cup](03_cnn.ipynb)
- 3.5 [One MLflow run](04_mlflow.ipynb)
- 3.6 [Churn tab and image tab](app.py)
- 3.7 [Why this model](why_this_model.md)
- [Column notes](data/DATA_DICTIONARY.md)

Run the commands on this page from `04-deep-learning` (this folder). Tool versions are pinned in the repo `mise.toml`. This folder has its own `pyproject.toml`. Do not edit Day 1 or Day 2.

## Timetable

Bangkok time. Breaks are 10:30–10:45 and 14:30–14:45. Lunch is 12:00–13:00.

| Time | Block |
| --- | --- |
| 09:00–10:30 | Neural net, then `MLPClassifier` on churn. Did it beat the random forest? |
| 10:45–12:00 | Learning rate, scaling, early stopping, learning curve. Train score versus validation score. |
| 13:00–14:30 | OpenCV, then the paper convolution, then a small CPU CNN. |
| 14:45–15:30 | Grocery Image Cup. Add the image-upload tab. |
| 15:30–16:00 | Open `mlflow ui`, find the one logged run, and share the memo. |

## What you build

A validation ROC-AUC for the forest and for one MLP, a short note on the train-versus-validation gap, one CNN with a held-out prediction file, a form with a churn tab and a product-image tab, and one MLflow run.

## Objectives

By the end of this workshop, you can:

- Say what a layer, a loss, and a learning rate are doing in `MLPClassifier`.
- Compare that network with the Day 1 random forest on validation ROC-AUC.
- Point at a learning rate or a missing scaler that hurts the validation score.
- Resize a grayscale image and run an edge filter.
- Train a two-layer CNN on CPU and read its validation accuracy.
- Upload an image to the form and read the class it returns.
- Find that run in the local MLflow UI.

## In the room

These games have no kit beyond the grid in [02_opencv.ipynb](02_opencv.ipynb). They are the picture for the notebooks.

Humans as a neural net. Three people stand in a line. The first hears a number of orders. Each person passes 1 if the number they received is 3 or more, and 0 otherwise. The last person says churn or stay. Then change the middle person's rule and send the same number through again.

Blindfold hill. One person is blindfolded on a line of tape. They take a step. A partner says only "higher" or "lower". A huge step jumps over the bottom of the hill. A tiny step takes all morning. That step size is the learning rate.

Memorize versus understand. Read three training customers and the label, and ask the room to recite them. Then describe a new customer the room has not heard. Reciting the three is the train score. The new customer is the validation score.

Humans as convolution filters. One person slides the 2 by 2 kernel in the OpenCV notebook across the pixel grid, one pixel at a time, and writes the sum. That sum is one output pixel.

## The data

Churn uses the Day 1 file, `01-supervised-learning/data/fresh_and_good_customers.csv`. This folder reads it and does not write it. The split is the Day 1 split: seed 42, 1440 train, 480 validation, 480 test. Leave the test rows unused.

Product images are Fashion-MNIST when `data/fashion/` contains the four IDX files. Those files are not in git. The license is MIT, Zalando SE, 2017. See [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md). If the files are missing, the notebooks use `sklearn.datasets.load_digits`. There is no download for digits.

The cup's held-out labels are not in this repo. The teaching team scores `models/cup_predictions.csv`.

## Expected outputs

Starter `hidden = (32,)` in `00_mlp_churn.ipynb`, validation ROC-AUC:

| Model | Train | Validation | Notes |
| --- | --- | --- | --- |
| Random forest | 0.959 | 0.776 | 80 trees, `min_samples_leaf` 5, seed 42 |
| MLP `(32,)` | 0.849 | 0.782 | Adam, 200 iterations, the loss has not settled |

The MLP is ahead on the validation rows by a little. The forest fits the train rows harder. On the latest Linux check the forest took 0.14 seconds and the MLP took 0.35 seconds.

`01_training.ipynb`, validation ROC-AUC unless noted:

| Setting | Train | Validation |
| --- | --- | --- |
| SGD, learning rate 0.001, 40 iterations | 0.680 | 0.656 |
| SGD, learning rate 0.01, 40 iterations | 0.790 | 0.772 |
| SGD, learning rate 1.0, 40 iterations | 0.871 | 0.748 |
| SGD 0.01, numeric columns scaled | 0.790 | 0.772 |
| SGD 0.01, numeric columns not scaled | 0.540 | 0.542 |
| Adam `(64, 64)`, no early stopping, 80 iterations | 0.913 | 0.756 |
| Adam `(64, 64)`, early stopping | 0.812 | 0.787 |

The gap on the "no early stopping" row, using these three-decimal scores, is 0.913 − 0.756 = 0.157. Early stopping stops at 27 iterations. The unscaled run stops at 13 iterations.

Learning curve, train rows only, 3-fold ROC-AUC:

| Train rows | Train | CV |
| --- | --- | --- |
| 240 | 0.653 | 0.618 |
| 480 | 0.693 | 0.669 |
| 720 | 0.729 | 0.704 |
| 960 | 0.750 | 0.728 |

`02_opencv.ipynb` draws a green square. The edge sum is 15300. The resized image has 256 bright pixels. If the webcam cell prints `No camera`, leave it.

`03_cnn.ipynb`, learning rate 0.001. With Fashion-MNIST files present, the fit uses the full seeded split: 42,000 train rows, 14,000 validation rows, and 14,000 held-out rows, for 12 epochs. On the Linux check that validation accuracy was 0.890, the train accuracy was 0.907, the held-out accuracy was 0.887, and the fit took 19.79 seconds. The notebook prints the validation accuracy as 0.89. A second fit in `04_mlflow.ipynb` took 21.04 seconds, so the time moves a little. Without the Fashion-MNIST files, digits use every row for 40 epochs. On the Linux check that validation accuracy was 0.953, the train accuracy was 0.976, the held-out accuracy was 0.942, and the fit took 0.75 seconds. Your laptop will differ. The score is the number to match for the data you actually loaded. The notebook prints `source`.

## Before class

Use the Day 1 install for mise 2026.11.3, uv 0.13.7, and Python 3.15.2. The Linux setup was tested for this day. The macOS and Windows commands for the notebook environment were not retested.

PyTorch 2.15.0 publishes a cp315 wheel for this Python. This project installs it from the CPU index, `https://download.pytorch.org/whl/cpu`, so a CUDA build is not selected. On Linux and Windows the wheel name ends in `+cpu`. On macOS Apple Silicon the CPU index serves `torch-2.15.0-cp315-cp315-macosx_14_0_arm64.whl` without that suffix. The Linux x86_64 CPU wheel downloaded here is 209,534,469 bytes.

`opencv-python` 5.0.0.93 ships `cp37-abi3` wheels, so the same wheel runs on Python 3.15 on Linux, macOS, and Windows. The project uses that package, not `opencv-python-headless`. The OpenCV docs say the headless builds omit GUI calls such as `cv2.imshow` and are the ones to pick when you do not need a camera preview. This day keeps the desktop package because the webcam cell calls `VideoCapture`. The notebook still skips that cell when no camera opens, and it never calls `imshow`.

## Create the environment

From this folder, after mise is on your `PATH`:

```sh
mise install
uv sync --frozen
PYTHONPATH=lib uv run python -m ipykernel install --user --name fresh-and-good-day3 --display-name "Python (fresh-and-good-day3)"
```

The repo `mise.toml` sets `PYTHONPATH` to `01-supervised-learning/lib`. Leave that line alone so Day 1 imports still work. For Day 3 commands, set `PYTHONPATH=lib` as the Makefile does. The notebooks put this folder's `lib` first on `sys.path`.

MLflow 3.18 keeps the filesystem tracking store only if you opt in. The notebooks and the Makefile set `MLFLOW_ALLOW_FILE_STORE=true`. Runs are files under `tracking/`. Nothing is uploaded. `tracking/` is gitignored. This day does not register a model.

`uv.lock` installs PyYAML 6.0.3 from source because that release has no cp315 wheel. If the build fails, check that your laptop has a C compiler.

Optional, when the room has a network:

```sh
PYTHONPATH=lib uv run python -m images --download
```

Check the image files, or the digits fallback:

```sh
PYTHONPATH=lib uv run python -m images --check
```

`make setup`, `make check`, `make notebook`, `make app`, and `make test` run the same steps. The tests use digits, so they do not need the download.

## Open the notebooks

From this folder:

```sh
uv run python -m jupyter notebook
```

If the kernel menu lists more than one Python, choose Python (fresh-and-good-day3). Run the notebooks in order. An empty `debrief` is the blank. The code around it still runs.

1. `00_mlp_churn.ipynb`
2. `01_training.ipynb`
3. `02_opencv.ipynb`
4. `03_cnn.ipynb`
5. `04_mlflow.ipynb`

## MLflow

After `04_mlflow.ipynb`:

```sh
MLFLOW_ALLOW_FILE_STORE=true uv run python -m mlflow ui --backend-store-uri "file://${PWD}/tracking" --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The experiment is `grocery-image`. You should see the learning rate, the validation accuracy, and `confusion.csv`.

The file store is in maintenance in MLflow 3.18. The project sets `MLFLOW_ALLOW_FILE_STORE` because this class keeps a folder of files and does not run a database. See the [backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) page. This day uses `mlflow.set_tracking_uri`, `mlflow.log_param`, `mlflow.log_metric`, and `mlflow.log_artifact`. It does not call `mlflow.register_model`.

## Open the form

After notebook 00 and notebook 03 have written `models/` (that folder is gitignored):

```sh
uv run python -m streamlit run app.py
```

The Churn tab switches between the random forest and the MLP. The Product image tab uploads a picture and prints a class. Fashion-MNIST classes are clothing names. The digits fallback uses `"0"` through `"9"`.

Load only a file you fit in class. A joblib file, and a `torch.save` file, can run code when they are loaded.

## Fill in the memo

`why_this_model.md` has three blanks: accuracy, explainability, and cost. At 15:30, read one sentence out loud. A wrong clothing label in this exercise is a stand-in for a wrong product label at the desk.

## For the teaching team

Completed notebooks and the held-out label file stay with the teaching team. They are not in this repo.

```sh
make test
python score_cup.py models/cup_predictions.csv path/to/cup_labels.csv
```

`lib/cup.py` defines `instructor_labels`. Call it for the same source the room used (`fashion` or `digits`) and save that frame. Do not commit it.

Before the room, on a machine that has a network, copy two things onto the stick you already use. There is no USB tool in this repo.

Fashion-MNIST, from this folder:

```sh
PYTHONPATH=lib uv run python -m images --download
```

Copy `data/fashion/`. The four files are the ones named in the data dictionary.

The PyTorch CPU wheel is large. `uv sync --frozen` fills the uv cache for this machine. The Linux x86_64 CPU wheel for torch 2.15.0 is the 209,534,469-byte file from `https://download.pytorch.org/whl/cpu`. Do not commit the wheel or the IDX files.

## Library versions

| Package | Version | Where |
| --- | --- | --- |
| mise | 2026.11.3 | repo `mise.toml` |
| uv | 0.13.7 | repo `mise.toml` |
| Python | 3.15.2 | this folder. PyTorch 2.15.0 has a cp315 wheel, so this day stays on the course Python. |
| numpy | 2.5.4 | notebooks |
| pandas | 3.1.1 | notebooks |
| scikit-learn | 1.9.1 | morning MLP, digits fallback, confusion matrix |
| mlflow | 3.18.0 | local file store, one CNN run |
| opencv-python | 5.0.0.93 | load, resize, filter, optional webcam |
| torch | 2.15.0+cpu | Linux and Windows CPU wheel. macOS Apple Silicon wheel is `2.15.0` from the same CPU index. |
| streamlit | 1.67.0 | form |
| jupyter | 1.1.1 | notebook server |
| ipykernel | 7.4.0 | kernel `fresh-and-good-day3` |

The automated checks also used pytest 9.2.0, nbclient 0.11.0, and nbformat 5.10.4. MLflow pulls matplotlib 3.11.2. The notebooks do not call it. `torch.version.cuda` is `None` for the installed wheel.
