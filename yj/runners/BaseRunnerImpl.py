import numpy.random
from random import seed
from yj import Shaper, FManager

import pandas as pd
from yj.runners import BaseRunner

class BaseRunnerImpl(BaseRunner.BaseRunner):

    def __init__(self, lg):
        from yj.environment import RANDOM_SEED
        self.lg = lg
        numpy.random.seed(RANDOM_SEED)
        seed(RANDOM_SEED)

    def run(self, actions, cuda):
        if "preprocess" in actions:
            self.preprocess()
        if "train" in actions:
            self.train(cuda)
        if "predict" in actions:
            self.predict()

    def preprocess(self):
        from yj.environment import CAL_DTYPES, DATA_FOLDER, CALENDAR_CSV_F, PRICE_DTYPE, \
            SELL_PRICE_CSV, NUM_COLS, CAT_COLS, SALES_DTYPE, SALES_TRAIN_VAL, DATE_FEATURES, \
            PREPROCESS_OBJ_FP

        csv_files = [DATA_FOLDER + "/" + CALENDAR_CSV_F, DATA_FOLDER + "/" + SELL_PRICE_CSV,
                     DATA_FOLDER + "/" + SALES_TRAIN_VAL]
        dtypes = [CAL_DTYPES, PRICE_DTYPE, SALES_DTYPE]

        usecols = [None, None, CAT_COLS + NUM_COLS]

        csv_dtype_use_tup = zip(csv_files, dtypes, usecols)

        # shadows events to their lead up
        # calendar = Shaper.apply_label_before(calendar, ['event_name_1', 'event_name_1', 'event_type_1', 'event_type_2'],
        #                                      14)

        # transform categorical features into integers
        cal_df, prices, sales_df = FManager._extract(csv_dtype_use_tup)

        cal_df["date"] = pd.to_datetime(cal_df["date"])

        # melt the categorical data with the label being sales data
        sales_df = pd.melt(sales_df,
                           id_vars=CAT_COLS,
                           value_vars=[col for col in sales_df.columns if col.startswith("d_")],
                           var_name="d",
                           value_name="sales")

        self.lg.debug("melted the sales data...")

        # Merge "ds" with "calendar" and "prices" dataframe
        sales_df = sales_df.merge(cal_df, on="d", copy=False)
        sales_df = sales_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

        for featName, featFunc in DATE_FEATURES.items():
            if featName in sales_df.columns:
                sales_df[featName] = sales_df[featName].astype("int16")
            else:
                sales_df[featName] = getattr(sales_df["date"].dt, featFunc).astype("int16")

        self.lg.debug("creating the lag features...")

        # creating the features for 7 and 28 day moving average sales
        dayLags = [7, 28]
        lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]

        for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
            sales_df[lagSalesCol] = sales_df[["id", "sales"]].groupby("id")["sales"].shift(dayLag)

        windows = [7, 28]
        for window in windows:
            for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
                sales_df[f"rmean_{dayLag}_{window}"] = sales_df[["id", lagSalesCol]].groupby("id")[
                    lagSalesCol].transform(
                    lambda x: x.rolling(window).mean())

        # remove all rows with NaN value
        sales_df.dropna(inplace=True)

        cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id',
                    'month', 'year', 'event_name_1', 'event_name_2', 'event_type_1',
                    'event_type_2', 'week', 'quarter', 'mday']

        for cat_col in cat_cols:
            sales_df = Shaper.bottom_out_col(sales_df, cat_col)

        # define columns that need to be removed
        unusedCols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
        trainCols = sales_df.columns[~sales_df.columns.isin(unusedCols)]

        X_train = sales_df[trainCols]
        y_train = sales_df["sales"]
        del sales_df

        check_dfs = [X_train, y_train, trainCols, cal_df, prices]

        for df, obj_fp in zip(check_dfs, PREPROCESS_OBJ_FP):
            FManager.save(df, obj_fp)

        # create the test data as well
        Shaper.create_test()