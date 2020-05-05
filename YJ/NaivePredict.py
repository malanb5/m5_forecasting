import pickle, numpy as np, concurrent, tqdm, math
from YJ.Shaper import dropper, bin_columns
from YJ import Sharder

def pick_sales(pd_days, weekdays, r):
    sales = list()

    for i, row in weekdays.iterrows():
        day = row.loc['wday']
        pick = _get_sales_from_pd(pd_days[day], r =None, prob_choice="aggregate")
        sales.append(pick)

    return sales

def make_dow_pd():
    cal = pickle.load(open("../objs/eda_objs/0", "rb"))
    val = pickle.load(open("../objs/eda_objs/1", "rb"))
    ids = val['id']

    val = dropper(val, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    val = val.set_index(np.arange(0, 1913))

    cal = dropper(cal, ["date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2",
                                  "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])
    cal = cal.T

    pr_cal = cal[1914:]
    ori_cal = cal[:1914]

    bins = bin_columns(val, ori_cal, lr=.995)
    pickle.dump(bins, open("../objs/eda_objs/binned_dow_val.pkl", "wb"))

    bins = pickle.load(open("../objs/eda_objs/binned_dow_val.pkl", "rb"))

def get_sales_pd(pd_l_id, pr_cal):
    """
    gets the most likely outcomes from the probability distribution of a time series of days
    :param pd_l: tuple of the list of item's probability distributions and the id
    :param pr_cal: the probability of the calendar days to predict
    :return:
    """
    id, pd_l = pd_l_id

    prod_sales_pred = list()

    r = np.random

    for each_pd in tqdm(pd_l):
        sales = pick_sales(each_pd, pr_cal, r)
        prod_sales_pred.append(sales)

    return id, prod_sales_pred

def _get_sales_from_pd(pd_d, r=None, prob_choice="aggregate"):
    if prob_choice =="random":
        pick = r.uniform()

        # print("pick: %f"%pick)
        tally = 0
        keys = pd_d.keys()
        keys = list(keys)
        keys.sort()

        for k in keys:
            tally += pd_d[k]
            # print("tally, %f" % tally)
            if pick <= tally:
                return k

        return keys[len(keys) -1]
    elif prob_choice== "aggregate":
        tally = 0
        for k in pd_d.keys():
            tally += (pd_d[k] * k)

        return tally

    else:
        raise NotImplementedError()

def compare_sales(act_sales, predict_sales):
    print(act_sales)
    print(predict_sales)
    RMSE = 0
    N = len(predict_sales)
    sum = 0
    for i in range(N-30, N):
        sum += pow((act_sales['sales'][i]-predict_sales['sales'][i]),2)
    RMSE = math.sqrt((sum/N))

    print(RMSE)
    return RMSE

def naive_predict():

    pd_dow = pickle.load(open('../objs/eda_objs/norm_dow_val.pkl', "rb"))

    cal = pickle.load(open("../objs/eda_objs/0", "rb"))

    cal = dropper(cal, ["date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2",
                               "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])
    cal = cal.T

    pr_cal = cal[1914:]
    n = 2

    import time
    start_time = time.time()

    pd_dow_shards = Sharder.shard(pd_dow, n)

    pred_sales = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        future_sales = {executor.submit(get_sales_pd, (i, shard), pr_cal):
                            (i, shard) for i, shard in enumerate(pd_dow_shards)}

        for f in concurrent.futures.as_completed(future_sales):
            id, res = f.result()
            pred_sales[id] = res

    print("--- %s seconds ---" % (time.time() - start_time))

    pickle.dump(pred_sales, open('../objs/eda_objs/pred_sales.pkl', "wb"))