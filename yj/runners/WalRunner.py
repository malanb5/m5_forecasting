"""
Main runner class of the Walmart sales forecaster
"""

import argparse, logging

from yj import Logger
from yj.runners import NNRunner, LGBMRunner, ProphetRunner


class Main:

	@staticmethod
	def main():

		lg = Logger.init(level=logging.DEBUG)

		parser = argparse.ArgumentParser(description='predict sales data.')
		parser.add_argument('--algorithm', dest='algorithm', type=str, default="lgbm")
		parser.add_argument('--cuda', dest='cuda', type=bool, default=False)
		parser.add_argument('--actions', nargs="+", type=str)

		args = parser.parse_args()

		if args.algorithm == "prophet":
			ProphetRunner.prophet_predict(mt=False, lg=lg)
		elif args.algorithm == "lgbm":
			runner = LGBMRunner.LGBMRunner(lg=lg)
			runner.run(actions=args.actions, cuda=args.cuda)
		elif args.algorithm == "nn":
			runner = NNRunner.NNRunner(lg=lg)
			runner.run(actions=args.actions, cuda=args.cuda)
		else:
			raise Exception("please specify an algorithm to run eg. --algorithm lgbm")


Main.main()