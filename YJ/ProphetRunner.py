from fbprophet import Prophet
import pickle
import tqdm
from YJ.environment import *
from YJ.Sharder import find_shard_points
import concurrent

__all__=["prophet_predict"]

def _prophet_predict(i, df, future, Prophet):
    m =Prophet()
    m.add_country_holidays("US")
    m.fit(df)

    forecast = m.predict(future)

    return forecast["yhat"]



def _prophet_mt_predict(i, pob, start_i, fin_i, fut_dat):
    s_pred = list()
    start_i = 6098
    fin_i = 12196

    for i in tqdm(range(start_i, fin_i)):
        df = pob.loc[:, ["ds", i]]
        df.rename(mapper={i: 'y'}, axis=1, inplace=True)
        yhat = _prophet_predict(i, df, fut_dat, Prophet)
        s_pred.append(yhat)

    pickle.dump(s_pred, open(WORKING_DIR + '/eda_objs/s_predt_prophet_%d_%d.pkl' % (start_i, fin_i), 'wb'))

    return i, s_pred



def prophet_predict(mt=False, lg=None):

    eda_obs_fp = '/objs/eda_objs/'

    pob = pickle.load(open(WORKING_DIR + eda_obs_fp + 'timeseries_sales.pkl', "rb"))
    fut_dat = pickle.load(open(WORKING_DIR + eda_obs_fp + 'predict_cal', 'rb'))

    if not mt:
        s_pred = list()

        for i in tqdm(range(13500, len(pob.columns))):
            df = pob.loc[:, ["ds", i]]
            df.rename(mapper={i: 'y'}, axis=1, inplace=True)
            yhat = _prophet_predict(i, df, fut_dat, Prophet)
            s_pred.append(yhat)
            if i % 100 == 0:
                pickle.dump(s_pred,
                            open(WORKING_DIR + eda_obs_fp + 's_predt_prophet_holidays_4_20_20_13500_30000.pkl', 'wb'))

        pickle.dump(s_pred, open(WORKING_DIR + eda_obs_fp + 's_predt_prophet_holidays_4_20_20_13500_30000.pkl', 'wb'))

    elif (mt):
        import time
        start_time = time.time()
        n = 1
        pred_sales = [None] * (n + 1)

        tot_data_points = len(pob.columns) - 1
        shard_points = find_shard_points(tot_data_points, n)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            future_sales = {
                executor.submit(_prophet_mt_predict, i, pob, start_fin_tup[0], start_fin_tup[1], fut_dat):
                    (i, start_fin_tup[0], start_fin_tup[1]) for i, start_fin_tup in enumerate(shard_points)}

            for f in concurrent.futures.as_completed(future_sales):
                id, res = f.result()
                pred_sales[id] = res

        pickle.dump(pred_sales, open(WORKING_DIR + '/eda_objs/prophet_pred_sales_mt_18294_24392.pkl', "wb"))

        print("--- %s seconds ---" % (time.time() - start_time))
        exit(0)