import sklearn.metrics
import Shaper

def compare_fb_nn():
	"""
	compares the Neural Network score with the FB Prophet Score
	:return:
	"""
	fb_scores = Shaper.load("objects/s_predt_prophet_no_holiday_20_4_19_index_0.pkl")
	nn_scores = Shaper.load("objects/scores_nnpredict_np_array_2020-04-20_19-39.pkl")

	print(fb_scores)
	print(nn_scores)

	assert(len(fb_scores) == len(nn_scores))

	diffs = abs(fb_scores - nn_scores)

	print(diffs)

	loss = sklearn.metrics.mean_squared_error(fb_scores, nn_scores)

	print(loss)