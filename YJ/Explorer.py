"""
EDA methods
methods specifically for this project
"""

import sklearn.metrics
from YJ import Shaper, FManager
import pandas as pd, pickle, numpy as np

def compare_fb_nn():
	"""
	compares the Neural Network score with the FB Prophet Score
	:return:
	"""
	fb_scores = Shaper.load("eda_objs/s_predt_prophet_no_holiday_20_4_19_index_0.pkl")
	nn_scores = Shaper.load("eda_objs/scores_nnpredict_np_array_2020-04-20_19-39.pkl")

	print(fb_scores)
	print(nn_scores)

	assert(len(fb_scores) == len(nn_scores))

	diffs = abs(fb_scores - nn_scores)

	print(diffs)

	loss = sklearn.metrics.mean_squared_error(fb_scores, nn_scores)

	print(loss)

def make_str_cat_numeric():

    FManager.save(pd.read_csv("../data/calendar.csv"), "eda_objs/sales_calendar.pkl")

    cal = FManager.load("eda_objs/sales_calendar.pkl")

    cat_col = ['event_type_1', 'event_type_2', 'event_name_1', 'event_name_2']
    cal.fillna(0, inplace=True)
    cal.reset_index(inplace=True)
    cal.drop(['weekday'], axis=1, inplace=True)

    for cat in cat_col:
        print(cal.head()[cat])
        print(cal.tail()[cat])

    cal = Shaper._make_cat_numeric(cal, cat_col)
    pickle.dump(cal, open("../objs/eda_objs/sales_calendar_binned_events.pkl", "wb"))

def make_holidays_before():
    df_in_fp = "eda_objs/sales_calendar_binned_events.pkl"
    df_out_fp = "eda_objs/sales_calendar_binned_events_holidays_before_constant.pkl"
    df = FManager.load(df_in_fp)

    cat_col = ['event_type_1', 'event_type_2', 'event_name_1', 'event_name_2']
    df = FManager.apply_label_before(df, cat_col, 7)
    print(df)

    pickle.dump(df, open(df_out_fp, "wb"))

def make_labels_get_max_score():
    sales = pickle.load(open("../objs/eda_objs/CA_sales_cat.pkl", "rb"))
    cal = pickle.load(open("../objs/eda_objs/sales_calendar_binned_events_holidays_before_constant.pkl", "rb"))

    cal = cal.loc[1913:, :]

    sales.drop(['item_id', 'id', 'cat_id', 'store_id', 'item_id', 'dept_id', "state_id"], axis=1, inplace=True)

    train_df = pd.DataFrame(data={'snap_CA': cal['snap_CA'], 'wday': cal['wday'], 'month': cal['month'],
                                  'event_type_1': cal['event_type_1'], 'event_type_2': cal['event_type_2'],
                                  'event_name_1': cal['event_name_1'], 'event_name_2': cal['event_name_2']})

    sales = sales.T

    sales = sales.reset_index()

    # pick which one to choose the labels from
    labels = sales.loc[:, 0]

    counts = labels.value_counts()

    max_score = counts.size

    print(max_score)

    labels = np.asarray(labels)

    # convert to a the training
    train_mat = np.asarray(train_df)
    print(train_mat)
    print(len(train_mat))

    pickle.dump(train_mat, open(
        "../objs/eda_objs/np_matrix_snap_ca_wday_month_event_type_event_type_event_name_ca_forcast_1913_1970.pkl", "wb"))
    # pickle.dump(labels, open('eda_objs/np_labels_ca_wday_month_event_type_name_forcast_1913_1930.pkl', "wb"))

def create_prediction_data():
    import pandas as pd, numpy as np, datetime
    # %%

    calendar = Shaper.load("objs/calendar_20_4_20.pkl")
    prices = Shaper.load("objs/prices_20_4_20.pkl")
    trainCols = Shaper.load("objs/trainCols.pkl")

    # Last day used for training
    trLast = 1913
    # Maximum lag day
    maxLags = 57

    # Create dataset for predictions
    def create_ds():
        startDay = trLast - maxLags

        numCols = [f"d_{day}" for day in range(startDay, trLast + 1)]
        catCols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']

        dtype = {numCol: "float32" for numCol in numCols}
        dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

        ds = pd.read_csv("../data/sales_train_validation.csv",
                         usecols=catCols + numCols, dtype=dtype)

        for col in catCols:
            if col != "id":
                ds[col] = ds[col].cat.codes.astype("int16")
                ds[col] -= ds[col].min()

        for day in range(trLast + 1, trLast + 28 + 1):
            ds[f"d_{day}"] = np.nan

        ds = pd.melt(ds,
                     id_vars=catCols,
                     value_vars=[col for col in ds.columns if col.startswith("d_")],
                     var_name="d",
                     value_name="sales")

        ds = ds.merge(calendar, on="d", copy=False)
        ds = ds.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

        return ds

    def create_features(ds):
        dayLags = [7, 28]
        lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]
        for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
            ds[lagSalesCol] = ds[["id", "sales"]].groupby("id")["sales"].shift(dayLag)

        windows = [7, 28]
        for window in windows:
            for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
                ds[f"rmean_{dayLag}_{window}"] = ds[["id", lagSalesCol]].groupby("id")[lagSalesCol].transform(
                    lambda x: x.rolling(window).mean())

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

    fday = datetime.datetime(2016, 4, 25)
    te = create_ds()

    for tdelta in range(0, 28):
        day = fday + datetime.timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te['date'] >= day - datetime.timedelta(days=maxLags)) & (te['date'] <= day)].copy()
        create_features(tst)
        tst = tst.loc[tst['date'] == day, trainCols]
        print(tst)
        Shaper.save(tst, "objs/input_d_%d.pkl" % tdelta)

        # te.loc[te['date'] == day, "sales"] = alpha * m_lgb.predict(tst)  # magic multiplier by kyakovlev

def plot_best_features():
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import seaborn as sns

    columns = FManager.load("objs/X_train.pkl").columns

    clf = lgb.Booster(model_file="../models/model.lgb")

    # feature importance comes built in with lgb
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), columns)), columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 20))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))
    plt.title('LightGBM Features')
    plt.tight_layout()
    # plt.show()
    plt.savefig('images/lgbm_importances-01.png')