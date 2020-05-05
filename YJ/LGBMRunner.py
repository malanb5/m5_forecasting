import pandas as pd, numpy as np
from YJ import Shaper, FManager
from YJ.environment import LGBM_PARAMS, MODEL_FP, ALPHAS, WEIGHTS, TR_LAST, MAX_LAGS, FDAY
from YJ.environment import SUBMISSION_FP, DS_FPS, CAT_FEATURES
import lightgbm as lgb
import gc
from datetime import timedelta

def lgbm_preprocess(lg):

    # the calendar data types
    calendarDTypes = {"event_name_1": "category",
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
                      'snap_WI': 'float32'}

    calendar = pd.read_csv("data/calendar.csv",
                           dtype=calendarDTypes)
    lg.debug("loaded calendar data")

    # shadows events to their lead up
    calendar = Shaper.apply_label_before(calendar, ['event_name_1', 'event_name_1', 'event_type_1', 'event_type_2'],
                                         14)

    calendar["date"] = pd.to_datetime(calendar["date"])

    # transform categorical features into integers
    for col, colDType in calendarDTypes.items():
        if colDType == "category":
            calendar[col] = calendar[col].cat.codes.astype("int16")
            calendar[col] -= calendar[col].min()

    # data types for "sell_prices.csv"
    priceDTypes = {"store_id": "category",
                   "item_id": "category",
                   "wm_yr_wk": "int16",
                   "sell_price": "float32"}

    prices = pd.read_csv("data/sell_prices.csv",
                         dtype=priceDTypes)
    lg.debug("loaded price data")
    # Transform categorical features into integers
    for col, colDType in priceDTypes.items():
        if colDType == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    lg.debug("converted price data...")
    ### *sales_train_validation.csv*

    firstDay = 250
    lastDay = 1913

    # Use x sales days (columns) for training
    numCols = [f"d_{day}" for day in range(firstDay, lastDay + 1)]

    # Define all categorical columns
    catCols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']


    # Define the correct data types for "sales_train_validation.csv"
    dtype = {numCol: "float32" for numCol in numCols}
    dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

    # Read csv file
    ds = pd.read_csv("data/sales_train_validation.csv",
                     usecols=catCols + numCols, dtype=dtype)

    # FEATURE ENGINEERING
    lg.debug("loading in sales train validation data...")
    # Transform categorical features into integers
    for col in catCols:
        if col != "id":
            ds[col] = ds[col].cat.codes.astype("int16")
            ds[col] -= ds[col].min()

    # melt the categorical data with the label being sales data
    ds = pd.melt(ds,
                 id_vars=catCols,
                 value_vars=[col for col in ds.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

    lg.debug("melted the sales data...")

    # Merge "ds" with "calendar" and "prices" dataframe
    ds = ds.merge(calendar, on="d", copy=False)
    ds = ds.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    dateFeatures = {"wday": "weekday",
                    "week": "weekofyear",
                    "month": "month",
                    "quarter": "quarter",
                    "year": "year",
                    "mday": "day"}

    for featName, featFunc in dateFeatures.items():
        if featName in ds.columns:
            ds[featName] = ds[featName].astype("int16")
        else:
            ds[featName] = getattr(ds["date"].dt, featFunc).astype("int16")

    lg.debug("creating the lag features...")

    # creating the features for 7 and 28 day moving average sales
    dayLags = [7, 28]
    lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]
    for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
        ds[lagSalesCol] = ds[["id", "sales"]].groupby("id")["sales"].shift(dayLag)

    windows = [7, 28]
    for window in windows:
        for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
            ds[f"rmean_{dayLag}_{window}"] = ds[["id", lagSalesCol]].groupby("id")[lagSalesCol].transform(
                lambda x: x.rolling(window).mean())

    lg.debug("cleaning up...")

    # remove all rows with NaN value
    ds.dropna(inplace=True)

    # define columns that need to be removed
    unusedCols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
    trainCols = ds.columns[~ds.columns.isin(unusedCols)]

    X_train = ds[trainCols]
    y_train = ds["sales"]
    del ds
    objs_fps = ["objs/X_train_LGBmodel.pkl", "objs/y_train_LGBmodel.pkl", "objs/trainCols_LGBmodel.pkl",
                "objs/calendarCols_LGBmodel.pkl", "objs/priceCols_LGBmodel.pkl"]
    check_dfs = [X_train, y_train, trainCols, calendar, prices]

    for df, obj_fp in zip(check_dfs, objs_fps):
        FManager.save(df, obj_fp)


def lgbm_get_model(lg, X_train, y_train, isTrained=False):


    if not isTrained:
        # define categorical features
        validInds = np.random.choice(X_train.index.values, 2_000_000, replace=False)
        trainInds = np.setdiff1d(X_train.index.values, validInds)

        trainData = lgb.Dataset(X_train.loc[trainInds], label=y_train.loc[trainInds],
                                categorical_feature=CAT_FEATURES, free_raw_data=False)
        validData = lgb.Dataset(X_train.loc[validInds], label=y_train.loc[validInds],
                                categorical_feature=CAT_FEATURES, free_raw_data=False)

        lg.debug("cleaning up necessarily data structures...")
        del X_train, y_train, validInds, trainInds
        gc.collect()

        lg.debug("training lgb model...")

        # Train LightGBM model
        m_lgb = lgb.train(LGBM_PARAMS, trainData, valid_sets=[validData], verbose_eval=20)

        # # Save the model
        m_lgb.save_model(MODEL_FP)

    else:
        # loading the pre-trained model
        m_lgb = lgb.Booster(model_file=MODEL_FP)

    return m_lgb


def lgbm_predict(lg, prices, calendar, trainCols, m_lgb):
    # PREDICTIONS
    lg.debug("making predictions...")

    for icount, (alpha, weight) in enumerate(zip(ALPHAS, WEIGHTS)):
        te = Shaper.create_ds(TR_LAST, MAX_LAGS, calendar, prices)
        cols = [f"F{i}" for i in range(1, 29)]

        for tdelta in range(0, 28):
            day = FDAY + timedelta(days=tdelta)
            lg.debug("%s, %s" % (tdelta, day))
            tst = te[(te['date'] >= day - timedelta(days=MAX_LAGS)) & (te['date'] <= day)].copy()
            Shaper.create_features(tst)
            tst = tst.loc[tst['date'] == day, trainCols]
            te.loc[te['date'] == day, "sales"] = alpha * m_lgb.predict(tst)  # magic multiplier by kyakovlev

        te_sub = te.loc[te['date'] >= FDAY, ["id", "sales"]].copy()
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
        te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
        te_sub.fillna(0., inplace=True)
        te_sub.sort_values("id", inplace=True)
        te_sub.reset_index(drop=True, inplace=True)
        te_sub.to_csv(f"submission_{icount}.csv", index=False)

        if icount == 0:
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols] * weight

        lg.debug(icount, alpha, weight)

    lg.debug("creating csv file submissions")
    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv(SUBMISSION_FP, index=False)

def lgbm_run(lg):
    np.random.seed(777)

    X_train, y_train, trainCols, calendar, prices = FManager.load_objs(DS_FPS)

    m_lgb = lgbm_get_model(lg, X_train, y_train, isTrained=True)

    lgbm_predict(lg, prices, calendar, trainCols, m_lgb)