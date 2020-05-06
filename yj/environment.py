"""
Environmental variables and constants
"""
import os
from datetime import datetime

WORKING_DIR = os.getcwd()

DATA_FOLDER = "data"
CALENDAR_CSV_F = "calendar.csv"
SALES_TRAIN_VAL = "sales_train_validation.csv"
SELL_PRICE_CSV = "sell_prices.csv"
OBJ_FOLDER = "eda_objs"

RANDOM_SEED=0
DAYS_TO_PREDICT = 28
F_DAY = 250
L_DAY = 1913

obj_names = ["eda_objs/1"]

# LOADING
DS_FPS = ["objs/X_train_LGBmodel.pkl", "objs/y_train_LGBmodel.pkl", "objs/trainCols_LGBmodel.pkl",
            "objs/calendarCols_LGBmodel.pkl", "objs/priceCols_LGBmodel.pkl"]

X_Y_FPS = ["objs/X_train_LGBmodel.pkl", "objs/y_train_LGBmodel.pkl"]

PRICE_CAL_TRAIN_DF_FP = [
    "objs/trainCols_LGBmodel.pkl",
    "objs/calendarCols_LGBmodel.pkl",
    "objs/priceCols_LGBmodel.pkl"
]

CAT_FEATURES = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + \
               ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

# Use x sales days (columns) for training
NUM_COLS = [f"d_{day}" for day in range(F_DAY, L_DAY + 1)]

# Define all categorical columns
CAT_COLS = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']

LGBM_MODEL_FP = "models/model_events_before.lgb"
SUBMISSION_FP = "submissions/submission_nn_20_5_5.csv"

# Last day used for training
TR_LAST = 1913

# Maximum lag day
MAX_LAGS = 57

FDAY = datetime(2016, 4, 25)

ALPHAS = [
    1.028, 1.023, 1.018
]

WEIGHTS = [1 / len(ALPHAS)] * len(ALPHAS)

sub = 0.

NN_MODEL_FP = 'models/wal_nn_2020-05-05_03-35.hdf5'

# model parameters
LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.0075,
    "max_bin":63,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "metric": ["auc"],
    'verbosity': 1,
    'num_iterations': 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100
    # 'device': 'gpu',
    # 'gpu_platform_id':0,
    # 'gpu_device_id':0

}

CAL_DTYPES={
   "event_name_1": "category",
  "event_name_2": "category",
  "event_type_1": "category",
  "event_type_2": "category",
  "weekday": "category",
  'wm_yr_wk': 'int16',
  "wday": "int16",
  "month": "int16",
  "year": "int16",
  "snap_CA": "float32",
  'snap_TX': 'float32',
  'snap_WI': 'float32'
}

PRICE_DTYPE = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32"
}

DATE_FEATURES = {
    "wday": "weekday",
    "week": "weekofyear",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
    "mday": "day"
}

PREPROCESS_OBJ_FP = [
    "objs/X_train.pkl",
    "objs/y_train.pkl",
    "objs/trainCols.pkl",
    "objs/calendarCols.pkl",
    "objs/priceCols.pkl"
]

# Define the correct data types for "sales_train_validation.csv"
SALES_DTYPE = {numCol: "float32" for numCol in NUM_COLS}
SALES_DTYPE.update({catCol: "category" for catCol in CAT_COLS if catCol != "id"})