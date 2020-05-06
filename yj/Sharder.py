"""
Shards files and other eda_objs
"""
import pickle
from .environment import WORKING_DIR

def shard(l, n):
		n_l = list()
		ln = len(l)

		shard_size = int(ln/n)
		tot_sharded = 0
		for i in range(n):
			n_l.append(l[i *(shard_size): (i+1) * (shard_size)])
			tot_sharded += shard_size

		for i in range(tot_sharded, ln):
			n_l[len(n_l) -1].append(l[i])

		tot_shards = 0
		for i in range(len(n_l)):
			tot_shards+= len(n_l[i])

		assert(tot_shards == ln)

		return n_l

def find_shard_points(tot_l, n):

	shard_size = int(tot_l / n)
	shard_points = []
	tally = 0
	for i in range(n):
		shard_points.append([i * shard_size, (i + 1) * shard_size])
		tally += shard_size

	shard_points[len(shard_points) - 1][1] = shard_points[len(shard_points) - 1][1] + (tot_l - tally)

	return shard_points


def shard_columns(pob, n):
	tot_data_points = len(pob.columns) - 1
	shard_points = find_shard_points(tot_data_points, n)

	for st_end in shard_points:
		pickle.dump(pob[pob.columns[st_end[0]: st_end[1]]],
					open(WORKING_DIR + '/eda_objs/date_sales_shard_%d_%d.pkl' % (st_end[0], st_end[1]), "wb"))
