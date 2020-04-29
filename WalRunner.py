import Parser, os, pandas as pd, logging, numpy as np,math
from tqdm import tqdm
import time, pickle
import concurrent.futures
from datetime import datetime
from fbprophet import Prophet
import os

class Environ:
	def __init__(self):
		self.WORKING_DIR = ""

env = Environ()

class WalRunner:

	def __init__(self, lg):
		self.lg = lg

	@staticmethod
	def make_dow_pd():
		cal = pickle.load(open("objects/0", "rb"))
		val = pickle.load(open("objects/1", "rb"))
		ids = val['id']

		val = WalRunner.dropper(val, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
		val = val.set_index(np.arange(0, 1913))

		cal = WalRunner.dropper(cal, ["date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2",
									  "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])
		cal = cal.T

		pr_cal = cal[1914:]
		ori_cal = cal[:1914]

		bins = WalRunner.bin_columns(val, ori_cal, lr=.995)
		pickle.dump(bins, open("objects/binned_dow_val.pkl", "wb"))

		bins = pickle.load(open("objects/binned_dow_val.pkl", "rb"))

	@staticmethod
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
			sales = WalRunner.pick_sales(each_pd, pr_cal, r)
			prod_sales_pred.append(sales)

		return id, prod_sales_pred

	@staticmethod
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

	@staticmethod
	def pick_sales(pd_days, weekdays, r):
		sales = list()

		for i, row in weekdays.iterrows():
			day = row.loc['wday']
			pick = WalRunner._get_sales_from_pd(pd_days[day], r =None, prob_choice="aggregate")
			sales.append(pick)

		return sales

	@staticmethod
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


	@staticmethod
	def make_predictions():

		pd_dow = pickle.load(open('objects/norm_dow_val.pkl', "rb"))

		cal = pickle.load(open("objects/0", "rb"))

		cal = WalRunner.dropper(cal, ["date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2",
								   "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])
		cal = cal.T

		pr_cal = cal[1914:]
		n = 2

		import time
		start_time = time.time()

		pd_dow_shards = WalRunner.shard(pd_dow, n)

		pred_sales = [None] * n
		with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
			future_sales = {executor.submit(WalRunner.get_sales_pd, (i, shard), pr_cal):
								(i, shard) for i, shard in enumerate(pd_dow_shards)}

			for f in concurrent.futures.as_completed(future_sales):
				id, res = f.result()
				pred_sales[id] = res

		print("--- %s seconds ---" % (time.time() - start_time))

		pickle.dump(pred_sales, open('objects/pred_sales.pkl', "wb"))

		exit(0)

	@staticmethod
	def make_big_table():

		cal = pickle.load(open(env.WORKING_DIR + "/objects/0", 'rb'))
		sales = pickle.load(open(env.WORKING_DIR + "/objects/WI_sales_cat.pkl", 'rb'))
		prices = pickle.load(open(env.WORKING_DIR + "/objects/2", 'rb'))

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
					item_id = id[: (n_char -2)]
					store_id = id[(n_char-1):]

				n_char +=1

			item_prices.append(prices.loc[(prices['store_id'] == store_id) & (prices['item_id'] == item_id)])

		print(len(item_prices))

		pickle.dump(item_prices, open(env.WORKING_DIR + "/objects/item_prices_wi.pkl", "wb"))

	@staticmethod
	def prophet_shape():
		# states =["CA", "WI", "TX"]
		# for st in states:
		# 	pob = pickle.load(open(env.WORKING_DIR + "/objects/%s_sales_cat.pkl" %(st), "rb"))
			# WalRunner.make_cat_numeric(st)

		# pred = pickle.load(open("objects/s_predt_prophet_no_holiday_20_4_19.pkl", "rb"))
		WalRunner.make_big_table()

		exit(0)

		# pob = pickle.load(open(env.WORKING_DIR + "/objects/CA_sales_cat.pkl", "rb"))
		# cat_col = ['cat_id', 'store_id', 'dept_id']

		exit(0)

		cal = pickle.load(open(env.WORKING_DIR + "/objects/0", 'rb'))

		titles = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

		pob = WalRunner.dropper(pob, titles)



		cal = WalRunner.dropper(cal, ["wm_yr_wk", "weekday", "d", "wday", "month", "year", "event_name_1", "event_name_2",
								   "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])

		cal = cal.T
		cal.rename(mapper={"date":"ds"}, axis=1, inplace=True)

		pr_cal = cal[1914:]
		ori_cal = cal[:1914]
		pickle.dump(ori_cal, open(env.WORKING_DIR + '/objects/ori_cal', 'wb'))
		pickle.dump(pr_cal, open(env.WORKING_DIR + '/objects/predict_cal', 'wb'))

		exit(0)


		ts_s =pd.merge(ori_cal, pob, left_on='d', right_on=pob.index)

		ts_s = WalRunner.dropper(ts_s, ['d', 'wday'])
		ts_s.rename(mapper={'date':"ds"}, inplace=True)

		pickle.dump(ts_s.T, open(env.WORKING_DIR + '/objects/timeseries_sales.pkl', 'wb'))

	@staticmethod
	def _prophet_predict(i,df, future):
		m =Prophet()
		m.add_country_holidays("US")
		m.fit(df)

		forecast = m.predict(future)

		return forecast["yhat"]

	@staticmethod
	def _prophet_mt_predict(i, pob, start_i, fin_i, fut_dat):
		s_pred = list()
		start_i = 6098
		fin_i = 12196

		for i in tqdm(range(start_i, fin_i)):
			df = pob.loc[:, ["ds", i]]
			df.rename(mapper={i: 'y'}, axis=1, inplace=True)
			yhat = WalRunner._prophet_predict(i, df, fut_dat)
			s_pred.append(yhat)

		pickle.dump(s_pred, open(env.WORKING_DIR + '/objects/s_predt_prophet_%d_%d.pkl'%(start_i, fin_i), 'wb'))

		return i, s_pred

	@staticmethod
	def prophet_predict(mt=False):
		pob = pickle.load(open(env.WORKING_DIR + '/objects/timeseries_sales.pkl', "rb"))
		fut_dat = pickle.load(open(env.WORKING_DIR + '/objects/predict_cal', 'rb'))

		if not mt:
			s_pred=list()

<<<<<<< HEAD

			for i in tqdm(range(13500, len(pob.columns))):
=======
			for i in tqdm(range(len(pob.columns) -1)):
>>>>>>> f5bf791b2d26d0c65b4ce5e010f90166134e1444
				df = pob.loc[:, ["ds", i]]
				df.rename(mapper={i:'y'}, axis =1, inplace=True)
				yhat =WalRunner._prophet_predict(i, df, fut_dat)
				s_pred.append(yhat)
				if i % 100 == 0:
					pickle.dump(s_pred, open(env.WORKING_DIR + '/objects/s_predt_prophet_holidays_4_20_20_13500_30000.pkl', 'wb'))

			pickle.dump(s_pred, open(env.WORKING_DIR + '/objects/s_predt_prophet_holidays_4_20_20_13500_30000.pkl', 'wb'))

		elif(mt):
			import time
			start_time = time.time()
			n = 1
			pred_sales = [None] * (n +1)

			tot_data_points = len(pob.columns) - 1
			shard_points = WalRunner.find_shard_points(tot_data_points, n)

			with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
				future_sales = {executor.submit(WalRunner._prophet_mt_predict, i, pob, start_fin_tup[0], start_fin_tup[1], fut_dat):
									(i, start_fin_tup[0], start_fin_tup[1]) for i, start_fin_tup  in enumerate(shard_points)}

				for f in concurrent.futures.as_completed(future_sales):
					id, res = f.result()
					pred_sales[id] = res

			pickle.dump(pred_sales, open(env.WORKING_DIR + '/objects/prophet_pred_sales_mt_18294_24392.pkl', "wb"))

			print("--- %s seconds ---" % (time.time() - start_time))
			exit(0)

class Main:

	@staticmethod
	def main(dir = None):
		if dir is None:
			env.WORKING_DIR = os.getcwd()
		else:
			env.WORKING_DIR = dir

		WalRunner.prophet_predict(mt=False)

		exit(0)

		# 1 turn into a probability density function with learning rate based on the day of the week
		# make predictions based upon that PDF
		# TODO: cut off based on when the product was introduced

#
Main.main(os.getcwd())