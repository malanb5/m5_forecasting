"""
Generic data shaping methods
"""

import tqdm, pickle, pandas as pd
import numpy as np
from src import Timer

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

def load(fp):
    return pickle.load(open(fp, "rb"))

def save(obj, fp):
    pickle.dump(obj, open(fp, "wb"))

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

