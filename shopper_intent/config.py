"""Project-wide constants: paths, column groups and the random seed.

Everything that more than one module needs to agree on lives here, so the
training script, the notebook and the Streamlit app can never drift apart.
"""

from pathlib import Path

RNG = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

RAW_DATA_PATH = DATA_DIR / "online_shoppers_intention.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"

METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
)

# ---------------------------------------------------------------- columns
TARGET = "Revenue"
CLASS_NAMES = ["No purchase", "Purchase"]  # index == encoded label

# page-visit counters and time spent on each page group: all heavily right-skewed
COUNT_COLS = ["Administrative", "Informational", "ProductRelated"]
DURATION_COLS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
# Google Analytics metrics; already bounded rates, PageValues is skewed money
RATE_COLS = ["BounceRates", "ExitRates"]
VALUE_COLS = ["PageValues"]
SPECIAL_DAY_COL = "SpecialDay"

# integer codes that are really labels (browser id 2 is not "twice" browser id 1)
CODED_CATEGORICAL_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]
TEXT_CATEGORICAL_COLS = ["Month", "VisitorType"]
BOOL_COLS = ["Weekend"]

# all 17 input features, in the same order as the original UCI csv
FEATURE_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
