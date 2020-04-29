import matplotlib.pyplot as plt

class Plotter:

	@staticmethod
	def scatter(x, y, alpha):
		plt.scatter(x, y, alpha=alpha)
		plt.show()

	@staticmethod
	def plotDf(df, fig_name):
		labels = []
		df.drop(columns=["index"], inplace=True)

		for i, (name, row) in enumerate(df.iterrows()):
			if name != "index":
				plt.plot(row)
				labels.append(name)

		plt.legend(labels)
		plt.savefig("figures/%s"%(fig_name))
		plt.show()