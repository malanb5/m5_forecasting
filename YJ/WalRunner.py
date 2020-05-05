"""
Main runner class of the Walmart sales forecaster
"""

import argparse, logging

from YJ import ProphetRunner, Logger, LGBMRunner, NNRunner

class WalRunner:

	def __init__(self, lg):
		self.lg = lg

class Main:

	@staticmethod
	def main():

		lg = Logger.init(level=logging.DEBUG)

		parser = argparse.ArgumentParser(description='predict sales data.')
		parser.add_argument('--algorithm', dest='algorithm', type=str)
		args = parser.parse_args()

		if args.algorithm == "prophet":
			ProphetRunner.prophet_predict(mt=False, lg=lg)
		elif args.algorithm == "lgbm":
			LGBMRunner.lgbm_run(lg=lg)
		elif args.algorithm == "nn_train":
			NNRunner.nn_run(lg=lg)
		elif args.algorithm =="nn_eval_predict":
			NNRunner.nn_make_eval_predict(lg=lg)
		else:
			raise Exception("please specify an algorithm to run eg. --algorithm lgbm")

		exit(0)

		# TODO: CI pipeline
		# TODO: Coordination framework/issue tracking/project management

Main.main()