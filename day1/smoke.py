import sys

import matplotlib
import numpy
import pandas
import sklearn
import streamlit

from day1.generate import customer_bytes
from day1.paths import data_csv
from day1.schema import COLUMNS, N_ROWS


def main() -> int:
    print(f"python {sys.version.split()[0]}")
    print(f"numpy {numpy.__version__}")
    print(f"pandas {pandas.__version__}")
    print(f"matplotlib {matplotlib.__version__}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"streamlit {streamlit.__version__}")
    path = data_csv()
    if not path.is_file():
        print(f"missing {path}")
        return 1
    if path.read_bytes() != customer_bytes():
        print(f"mismatch {path}")
        return 1
    print(f"rows {N_ROWS}")
    print(f"columns {', '.join(COLUMNS)}")
    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
