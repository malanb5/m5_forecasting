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

	@staticmethod
	def plot_history(plot_dict):
		# list all data in history
		print(plot_dict.history.keys())
		# summarize history for accuracy
		plt.plot(plot_dict.history['accuracy'])
		plt.plot(plot_dict.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# summarize history for loss
		plt.plot(plot_dict.history['loss'])
		plt.plot(plot_dict.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()