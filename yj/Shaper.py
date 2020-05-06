"""
Generic data shaping methods
"""

from tqdm import tqdm
import pickle, pandas as pd, numpy as np

def bottom_out_col(df, col_name):
    df[col_name] = df[col_name] - df[col_name].min()
    return df

def get_max(df, col_name):
    return df[col_name].max() + 1

# Create dataset for predictions
def create_ds(trLast, maxLags, calendar, prices):
    startDay = trLast - maxLags

    numCols = [f"d_{day}" for day in range(startDay, trLast + 1)]
    catCols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']

    dtype = {numCol: "float32" for numCol in numCols}
    dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

    ds = pd.read_csv("data/sales_train_validation.csv",
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

def create_features(lg, ds):
    dayLags = [7, 28]
    lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]

    for dayLag, lagSalesCol in tqdm(zip(dayLags, lagSalesCols)):
        ds[lagSalesCol] = ds[["id", "sales"]].groupby("id")["sales"].shift(dayLag)
    lg.debug(ds)

    windows = [7, 28]
    for window in tqdm(windows):
        for dayLag, lagSalesCol in tqdm(zip(dayLags, lagSalesCols)):
            ds[f"rmean_{dayLag}_{window}"] = ds[["id", lagSalesCol]].groupby("id")[lagSalesCol].transform(
                lambda x: x.rolling(window).mean())

    dateFeatures = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day"
    }

    for featName, featFunc in tqdm(dateFeatures.items()):
        if featName in ds.columns:
            ds[featName] = ds[featName].astype("int16")
        else:
            ds[featName] = getattr(ds["date"].dt, featFunc).astype("int16")

        ds[featName] -= ds[featName].min()

    cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday',
                'month', 'year', 'event_name_1', 'event_name_2', 'event_type_1',
                'event_type_2', 'week', 'quarter', 'mday']

    for cat in cat_cols:
        ds[cat] -= ds[cat].min()

    return ds

def bin_columns(val_df, cal_df, lr, col_index='wday'):
    """
    bins the columns according to a column
    :param val_df:
    :param cal_df:
    :param lr:
    :param col_index: the column index on which to bin from
    :return:
    """
    prod_pd = list()

    for c_i in tqdm(range(len(val_df.columns))):
        prod_sales = dict()
        c_lr = pow(lr, len(val_df))

        for day, day_sales in enumerate(val_df[c_i]):
            dow = cal_df[col_index][day]
            if dow not in prod_sales:
                prod_sales[dow] = dict()
                prod_sales[dow][day_sales] = 1 * c_lr

            else:
                if day_sales in prod_sales[dow]:
                    prod_sales[dow][day_sales] += (1 * c_lr)
                else:
                    prod_sales[dow][day_sales] = 1 * c_lr
            c_lr /=lr

        prod_pd.append(prod_sales)

    return prod_pd

def make_columns_from_first(df):
    # titles = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    titles_to_drop = ["id"]

    calendar = pickle.load(open("../objs/eda_objs/0", "rb"))
    first_col = {"sales": df.drop(titles_to_drop)}

    extra = pd.Series([0] * (1970 - 1914), index=["d_%d" % (i) for i in range(1914, 1970)])

    calendar = calendar.drop([i for i in range(1913, 1969)])
    first_col = first_col.append(extra)

def make_prob_dist(d, nested = False):

    d_l = dict()
    if nested:

        for key in d.keys():
            print(key)
            total = 0

            for inner_key in d[key]:
                total+= d[key][inner_key]

            for inner_key in d[key]:
                if key not in d_l:
                    d_l[key] = dict()
                    d_l[key][inner_key] = d[key][inner_key]/total
                else:
                    d_l[key][inner_key] = d[key][inner_key] / total

    else:
        total = 0
        for key in d.keys():
            total+=d[key]

        for key in d.keys():
            if key not in d_l:
                d_l[key] =d[key] /total
            else:
                d_l[key] = d[key] /total


    print(d_l)
    return d_l

def dropper(val_df, col_to_drop):
    val_df = val_df.drop(columns=col_to_drop)
    return val_df.T

def normalize_dict(binned_df_l):
    # normalize a binned list of dataframes

    for i, each_binned_df in tqdm(enumerate(binned_df_l)):
        for k in each_binned_df.keys():
            total = 0
            for k_i in each_binned_df[k].keys():
                total += each_binned_df[k][k_i]

            for k_i in each_binned_df[k].keys():
                each_binned_df[k][k_i] = each_binned_df[k][k_i]/total

        binned_df_l[i] = each_binned_df

    return binned_df_l

def binup(row, bins):

    if row not in bins:
        size_bins = len(bins)
        bins[row] = size_bins

        return size_bins
    else:
        return bins[row]

def _make_cat_numeric(df, cat_cols):
    for cat in cat_cols:
        bins = dict()
        df[cat] = df.apply(lambda x: binup(x[cat], bins), axis=1)
        print(bins)
    return df

def apply_label_before(df, cat, n_before):
    """
    applies a label to rows which appear before the event
    """
    for i, row in df.iterrows():
        if i >n_before:
            events = row.loc[cat]
            if any(events):
                for j in range(i -1, (i-n_before -1),  -1):
                    before_row = df.loc[j, :]
                    before_events = before_row.loc[cat]
                    if any(before_events):
                        break
                    else:
                        df.loc[j, cat] = events

                wk_bf = df.loc[i-n_before -1 : i, cat]

    return df

def one_hot_encode_column():
    """

    :return:
    """
    from yj.FManager import load
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    ds = load("objs/sales_train_validation.pkl")

    item_cat_id = np.expand_dims(np.asarray(ds['item_id']), axis=1)

    enc = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=False)
    enc.fit(item_cat_id)
    print(enc.categories_)

    item_cat_id = enc.transform([item_cat_id[0]])
    print(item_cat_id)

def generate_unified():
    from yj.environment import WORKING_DIR
    cal = pickle.load(open(WORKING_DIR + "/eda_objs/0", 'rb'))
    sales = pickle.load(open(WORKING_DIR + "/eda_objs/WI_sales_cat.pkl", 'rb'))
    prices = pickle.load(open(WORKING_DIR + "/eda_objs/2", 'rb'))

    item_set = sales['id'].apply(lambda x: x.replace("_validation", ""))
    print(item_set)
    sales.drop(["id", "item_id", 'dept_id', 'cat_id', 'store_id', "state_id"], axis=1, inplace=True)
    sales = sales.T
    merged = pd.merge(sales, cal, left_on=sales.index, right_on=cal["d"])
    merged.drop(['snap_TX', 'snap_CA', 'month', 'year', 'weekday'], axis=1, inplace=True)

    # print(merged.columns)
    # merged_first = merged.loc[:,
    # 			   [0, "date", "wm_yr_wk", "wday", "d", "event_name_1", "event_name_2", "event_type_1",
    # 				"event_type_2", "snap_TX"]]

    item_prices = []
    for id in tqdm(item_set):
        counter = 0
        n_char = 0

        for i in range(len(id)):
            if id[i] == "_":
                counter += 1
            if counter == 3:
                item_id = id[: (n_char - 2)]
                store_id = id[(n_char - 1):]

            n_char += 1

        item_prices.append(prices.loc[(prices['store_id'] == store_id) & (prices['item_id'] == item_id)])

    print(len(item_prices))

    pickle.dump(item_prices, open(WORKING_DIR + "/eda_objs/item_prices_wi.pkl", "wb"))

def make_cat_dtype(df, schema_dtypes, dtype):
    # transform categorical features into integers
    for col, colDType in schema_dtypes.items():
        if colDType == "category":
            df[col] = df[col].cat.codes.astype(dtype=dtype)
            df[col] -= df[col].min()

def create_test():
    from yj.environment import TR_LAST, MAX_LAGS, DATA_FOLDER, CALENDAR_CSV_F, SALES_TRAIN_VAL, SELL_PRICE_CSV, CAL_DTYPES,\
        PRICE_DTYPE, SALES_DTYPE, CAT_COLS, NUM_COLS, DATE_FEATURES
    from yj import FManager, Shaper

    startDay = TR_LAST - MAX_LAGS

    numCols = [f"d_{day}" for day in range(startDay, TR_LAST + 1)]

    dtype = {numCol: "float32" for numCol in numCols}
    catCols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

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

    for day in range(TR_LAST + 1, TR_LAST + 28 + 1):
        sales_df[f"d_{day}"] = np.nan

    cal_df["date"] = pd.to_datetime(cal_df["date"])


    # melt the categorical data with the label being sales data
    sales_df = pd.melt(sales_df,
                       id_vars=CAT_COLS,
                       value_vars=[col for col in sales_df.columns if col.startswith("d_")],
                       var_name="d",
                       value_name="sales")


    # Merge "ds" with "calendar" and "prices" dataframe
    sales_df = sales_df.merge(cal_df, on="d", copy=False)
    sales_df = sales_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    for featName, featFunc in DATE_FEATURES.items():
        if featName in sales_df.columns:
            sales_df[featName] = sales_df[featName].astype("int16")
        else:
            sales_df[featName] = getattr(sales_df["date"].dt, featFunc).astype("int16")

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
    unusedCols = ["d", "weekday"]
    trainCols = sales_df.columns[~sales_df.columns.isin(unusedCols)]

    FManager.save(sales_df[trainCols], "objs/X_test.pkl")