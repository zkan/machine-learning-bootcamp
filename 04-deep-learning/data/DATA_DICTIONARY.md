# Day 3 tables

Fresh and Good is the fictional Bangkok online grocery from Day 1. The churn models still read the Day 1 customer file. The image models use Fashion-MNIST, or scikit-learn digits when those files are absent.

## Customers

The churn models read `01-supervised-learning/data/fresh_and_good_customers.csv`. This folder does not copy or change that file. The columns are in the [Day 1 data dictionary](../../01-supervised-learning/data/DATA_DICTIONARY.md).

`churned` is the label. The split is the Day 1 split: seed 42, 1440 train, 480 validation, 480 test. Score the morning models on the validation rows. Leave the test rows unused.

## Fashion-MNIST

The files are not in git. `python -m images --download` fetches four gzip IDX files from the [Zalando research repo](https://github.com/zalandoresearch/fashion-mnist). The license is MIT, Zalando SE, 2017, in that repo's LICENSE file.

| File | What it is |
| --- | --- |
| `train-images-idx3-ubyte.gz` | 60,000 images, 28 by 28, grayscale |
| `train-labels-idx1-ubyte.gz` | 60,000 labels |
| `t10k-images-idx3-ubyte.gz` | 10,000 images |
| `t10k-labels-idx1-ubyte.gz` | 10,000 labels |

The class names, in label order, are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. They are clothing photos used here as a stand-in for a product photo. They are not grocery items.

The notebook joins train and test, then splits again with seed 42: 60 percent train, 20 percent validation, 20 percent held out. That is 42,000 train rows, 14,000 validation rows, and 14,000 held-out rows. The fit uses every row for 12 epochs.

## Digits fallback

If `data/fashion/` is missing, `load_dataset("auto")` uses `sklearn.datasets.load_digits`. That is 1,797 images, 8 by 8, labels 0 through 9. No download. The same 60/20/20 split is used, and the fit uses every row of each part for 40 epochs. Ink values are scaled from 0–16 to 0–255 so they match the Fashion-MNIST range.

The held-out labels are not written into this repo. `instructor_labels` in `lib/cup.py` rebuilds them for the teaching team.
