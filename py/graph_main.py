#import numpy as np
import argparse
import logging
from sklearn.datasets import make_circles
import numpy as np

from graph_models import CloudGraph, MMSBM

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--log', required=True,
					help = 'log level in {DEBUG, INFO, WARNING, ERROR, CRITICAL}')
	args = ap.parse_args()

	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	file_handler = logging.FileHandler('info.log', mode='w')
	file_handler.setLevel(getattr(logging, args.log))
	logger.addHandler(file_handler)
	logger.info("hello world")


	# ----------- cloud graph -----------

	"""
	X_cloud, labels = make_circles(n_samples=100, noise=0.1, factor=.2)

	Cloud_graph = CloudGraph(X_cloud, labels)

	Cloud_graph.generateMutualKNNGraph(10, sym='loose')

	Cloud_graph.showGraph()

	Cloud_graph.kmeansClustering()

	Cloud_graph.spectralClustering()
	"""

	# kmeans on PCD error = 44%
	# kmeans on adjacency matrix = 24%
	# spectral clustering = 0%

	# ----------- MMSBM -----------

	alpha = 0.3 * np.array([1,1,1]) #np.random.rand(3)
	nVertices = 100
	#bernouilli_matrix = np.diag(np.ones(len(alpha)))
	bernouilli_matrix = np.ones(len(alpha))*0.1+np.diag([1,1,1])*0.8
	sparsity_param = 0.
	print('\nalpha = ', alpha)
	print('\nbernouilli maxtrix = \n', bernouilli_matrix)
	print('\nsparsity parameter = \n', sparsity_param)
	
	my_graph_model = MMSBM(alpha, nVertices, bernouilli_matrix, sparsity_param)
	my_graph_model.generateGraph()
	my_graph_model.showGraph()

	my_graph_model.groundTruthClustering()

	my_graph_model.kmeansClustering()

	my_graph_model.getDegreesMatrix()

	my_graph_model.spectralClustering()

	# alpha=0.05 bernouilli = diag kmeans=0% spectral=0%
	# alpha=0.3 bernouilli = 0.1/0.9 kmeans=27.66% spectral=28.33
	# alpha=0.3 bernouilli = 0.1/0.9 kmeans=8% spectral=4.5%

