"""
Main runner class of the Walmart sales forecaster
"""
import traceback

import pandas as pd, numpy as np,math
from tqdm import tqdm
import pickle
import concurrent.futures
from YJ.environment import *
import argparse, logging
from YJ.FManager import load, save
from datetime import timedelta
import YJ.Shaper as Shaper
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, Model
from keras.optimizers import Adam
import gc
from keras.layers import Embedding, Add, Input, concatenate, SpatialDropout1D
from keras.layers import Flatten, Dropout, Dense, BatchNormalization, Concatenate
from tqdm.keras import TqdmCallback
import tensorflow as tf
from YJ.Timer import get_timestamp_str

class WalRunner:

	def __init__(self, lg):
		self.lg = lg

	@staticmethod
	def make_dow_pd():
		cal = pickle.load(open("../objs/eda_objs/0", "rb"))
		val = pickle.load(open("../objs/eda_objs/1", "rb"))
		ids = val['id']

		val = WalRunner.dropper(val, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
		val = val.set_index(np.arange(0, 1913))

		cal = WalRunner.dropper(cal, ["date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2",
									  "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])
		cal = cal.T

		pr_cal = cal[1914:]
		ori_cal = cal[:1914]

		bins = WalRunner.bin_columns(val, ori_cal, lr=.995)
		pickle.dump(bins, open("../objs/eda_objs/binned_dow_val.pkl", "wb"))

		bins = pickle.load(open("../objs/eda_objs/binned_dow_val.pkl", "rb"))

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
	def naive_predict():

		pd_dow = pickle.load(open('../objs/eda_objs/norm_dow_val.pkl', "rb"))

		cal = pickle.load(open("../objs/eda_objs/0", "rb"))

		cal = Shaper.dropper(cal, ["date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2",
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

		pickle.dump(pred_sales, open('../objs/eda_objs/pred_sales.pkl', "wb"))

	@staticmethod
	def _prophet_predict(i, df, future, Prophet):
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
			yhat = WalRunner._prophet_predict(i, df, fut_dat, Prophet)
			s_pred.append(yhat)

		pickle.dump(s_pred, open(WORKING_DIR + '/eda_objs/s_predt_prophet_%d_%d.pkl'%(start_i, fin_i), 'wb'))

		return i, s_pred

	@staticmethod
	def prophet_predict(mt=False, lg=None):
		from fbprophet import Prophet

		eda_obs_fp = '/objs/eda_objs/'

		pob = pickle.load(open(WORKING_DIR + eda_obs_fp + 'timeseries_sales.pkl', "rb"))
		fut_dat = pickle.load(open(WORKING_DIR + eda_obs_fp + 'predict_cal', 'rb'))

		if not mt:
			s_pred=list()

			for i in tqdm(range(13500, len(pob.columns))):
				df = pob.loc[:, ["ds", i]]
				df.rename(mapper={i:'y'}, axis =1, inplace=True)
				yhat = WalRunner._prophet_predict(i, df, fut_dat, Prophet)
				s_pred.append(yhat)
				if i % 100 == 0:
					pickle.dump(s_pred, open(WORKING_DIR + eda_obs_fp + 's_predt_prophet_holidays_4_20_20_13500_30000.pkl', 'wb'))

			pickle.dump(s_pred, open(WORKING_DIR + eda_obs_fp +'s_predt_prophet_holidays_4_20_20_13500_30000.pkl', 'wb'))

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

			pickle.dump(pred_sales, open(WORKING_DIR + '/eda_objs/prophet_pred_sales_mt_18294_24392.pkl', "wb"))

			print("--- %s seconds ---" % (time.time() - start_time))
			exit(0)

	@staticmethod
	def lgbm_preprocess():

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
		logging.debug("loaded calendar data")

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
		logging.debug("loaded price data")
		# Transform categorical features into integers
		for col, colDType in priceDTypes.items():
			if colDType == "category":
				prices[col] = prices[col].cat.codes.astype("int16")
				prices[col] -= prices[col].min()

		logging.debug("converted price data...")
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
		logging.debug("loading in sales train validation data...")
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

		logging.debug("melted the sales data...")

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

		logging.debug("creating the lag features...")

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

		logging.debug("cleaning up...")

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
			save(df, obj_fp)

	# Create dataset for predictions
	@staticmethod
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

	@staticmethod
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

	@staticmethod
	def lgbm_get_model(lg, X_train, y_train, isTrained= False):
		import lightgbm as lgb
		import gc

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

	@staticmethod
	def load_objs(objs_fps):
		dfs = []

		for obj_fp in objs_fps:
			dfs.append(load(obj_fp))

		return dfs


	@staticmethod
	def lgbm_predict(lg, prices, calendar, trainCols, m_lgb):
		# PREDICTIONS
		lg.debug("making predictions...")

		for icount, (alpha, weight) in enumerate(zip(ALPHAS, WEIGHTS)):
			te = WalRunner.create_ds(TR_LAST, MAX_LAGS, calendar, prices)
			cols = [f"F{i}" for i in range(1, 29)]

			for tdelta in range(0, 28):
				day = FDAY + timedelta(days=tdelta)
				lg.debug("%s, %s" % (tdelta, day))
				tst = te[(te['date'] >= day - timedelta(days=MAX_LAGS)) & (te['date'] <= day)].copy()
				WalRunner.create_features(tst)
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

	@staticmethod
	def lgbm_run(lg):

		np.random.seed(777)

		X_train, y_train, trainCols, calendar, prices  = WalRunner.load_objs(DS_FPS)

		m_lgb = WalRunner.lgbm_get_model(lg, X_train, y_train, isTrained=True)

		WalRunner.lgbm_predict(lg, prices, calendar, trainCols, m_lgb)

	@staticmethod
	def get_max(df, col_name):
		return df[col_name].max()+ 1

	@staticmethod
	def make_embedded_layer(shape, name, max_val, embed_n):
		input = Input(shape=shape, name=name)
		embed = Embedding(max_val, embed_n)(input)

		return input, embed

	@staticmethod
	def make_dense_input_layer(shape, name, n_d_layers, act_type):
		input = Input(shape=shape, name=name)
		d_l = Dense(n_d_layers, activation=act_type)(input)
		d_l = BatchNormalization()(d_l)

		return input, d_l

	@staticmethod
	def make_keras_input(df, cols, X = None):
		if X is None:
			X = {}
			for each_col in cols:
				X[each_col] = df[each_col]
		else:
			for each_col in cols:
				X[each_col] = df[each_col]
		return X

	@staticmethod
	def nn_train(lg):
		model_fp = 'models/wal_nn_%s.hdf5'% get_timestamp_str()
		ds = load("objs/X_train.pkl")
		y = load("objs/y_train.pkl")

		cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday',
					'month', 'year', 'event_name_1', 'event_name_2', 'event_type_1',
					'event_type_2', 'week', 'quarter', 'mday']
		cont_cols = ['sell_price', 'lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7',
					 'rmean_7_28', 'rmean_28_28']


		input_layers = []
		hid_layers = []
		n_embed_out = 64
		dense_n = 1000
		batch_size = 20000
		epochs = 5
		lr_init, lr_fin = 0.001, 0.0001
		exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
		steps = int(len(ds) / batch_size) * epochs
		lr_decay = exp_decay(lr_init, lr_fin, steps)

		for cat_col in cat_cols:
			max_cat = WalRunner.get_max(ds, cat_col)
			in_layer, embed_layer = WalRunner.make_embedded_layer([1], cat_col, max_cat, n_embed_out)
			input_layers.append(in_layer)
			hid_layers.append(embed_layer)

		fe = concatenate(hid_layers)
		s_dout = SpatialDropout1D(0.2)(fe)
		x = Flatten()(s_dout)

		con_layers = []

		for con_col in cont_cols:
			in_layer, embed_layer = WalRunner.make_dense_input_layer([1], con_col, n_embed_out, 'relu')
			input_layers.append(in_layer)
			con_layers.append(embed_layer)

		con_fe = concatenate(con_layers)

		x = concatenate([x, con_fe])
		x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
		x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
		outp = Dense(1, activation='sigmoid')(x)

		optimizer_adam = Adam(lr=0.001, decay=lr_decay)

		model = Model(inputs=input_layers, outputs=outp, name="wal_net")
		model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
		model.summary()

		X = WalRunner.make_keras_input(ds, cat_cols)
		X = WalRunner.make_keras_input(ds, cont_cols, X)

		model.fit(X, y, batch_size=batch_size, epochs=2, shuffle=True, verbose=0, callbacks=[TqdmCallback(verbose=2)])
		del ds, y, X
		gc.collect()
		model.save(model_fp)

	@staticmethod
	def nn_run(lg):
		device_type = "GPU"

		tf.config.set_soft_device_placement(True)
		tf.debugging.set_log_device_placement(True)

		gpus = tf.config.experimental.list_physical_devices('GPU')
		cpus = tf.config.experimental.list_physical_devices('CPU')

		if gpus and device_type == "GPU":
			for gpu in gpus:
				try:
					with tf.device(gpu.name.replace('physical_device:', '')):
						WalRunner.nn_train(lg)
				except RuntimeError as e:
					print(e)
					traceback.print_tb(e)
					traceback.print_stack(e)

class Main:
	@staticmethod
	def init_logger(level):
		import YJ.Timer as Timer
		import sys

		ds = Timer.get_timestamp_str()

		# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
		logPath = WORKING_DIR + "/logs"
		fileName = "%s"%ds


		logFormatter = logging.Formatter("%(asctime)s [%(name)-5.5s] [%(levelname)-5.5s]  %(message)s")
		rootLogger = logging.getLogger()

		fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
		fileHandler.setFormatter(logFormatter)
		rootLogger.addHandler(fileHandler)

		consoleHandler = logging.StreamHandler(sys.stdout)
		consoleHandler.setFormatter(logFormatter)
		rootLogger.addHandler(consoleHandler)
		rootLogger.setLevel(level)

		return rootLogger

	@staticmethod
	def main():

		lg = Main.init_logger(level=logging.DEBUG)

		parser = argparse.ArgumentParser(description='predict sales data.')
		parser.add_argument('--algorithm', dest='algorithm', type=str)
		args = parser.parse_args()

		if args.algorithm == "prophet":
			WalRunner.prophet_predict(mt=False, lg=lg)
		elif args.algorithm == "lgbm":
			WalRunner.lgbm_run(lg=lg)


		elif args.algorithm == "nn":
			WalRunner.nn_run(lg=lg)
		else:
			raise Exception("please specify an algorithm to run eg. --algorithm lgbm")

		exit(0)

		# TODO: CI pipeline
		# TODO: Coordination framework/issue tracking/project management

Main.main()