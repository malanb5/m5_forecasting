"""
Environmental variables and constants
"""
import os
from datetime import datetime

WORKING_DIR = os.getcwd()
DATA_FOLDER="data"
CALENDAR_F="calendar.csv"
SALES_TRAIN_VAL="sales_train_validation.csv"
SELL_PRICE="sell_prices.csv"

OBJ_FOLDER="eda_objs"


DAYS_TO_PREDICT = 28

obj_names=["eda_objs/1"]

# LOADING
DS_FPS = ["objs/X_train_LGBmodel.pkl", "objs/y_train_LGBmodel.pkl", "objs/trainCols_LGBmodel.pkl",
            "objs/calendarCols_LGBmodel.pkl", "objs/priceCols_LGBmodel.pkl"]

CAT_FEATURES = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + \
               ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

MODEL_FP = "models/model_events_before.lgb"
SUBMISSION_FP = "submissions/submission_lgbm_tree_20_5_3.csv"

# Last day used for training
TR_LAST = 1913
# Maximum lag day
MAX_LAGS = 57

FDAY = datetime(2016, 4, 25)
ALPHAS = [1.028, 1.023, 1.018]

WEIGHTS = [1 / len(ALPHAS)] * len(ALPHAS)

sub = 0.

# model parameters
LGBM_PARAMS = {
    "objective": "poisson",
    "metric": "rmse",
    "force_row_wise": True,
    "learning_rate": 0.075,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations': 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}