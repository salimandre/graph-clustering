import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations
import networkx as nx

class MMSBM:
	def __init__(self, alpha, nVertices, bernouilli_matrix, sparsity_param=0.):
		self.alpha=alpha
		self.K = len(self.alpha)
		self.nVertices = nVertices
		self.bernMat = bernouilli_matrix
		self.rho = sparsity_param
		properInit = (self.bernMat.shape[0] == self.bernMat.shape[1]) and (self.bernMat.shape[1] == self.K)
		if not properInit:
			raise Exception("bernouilli_matrix must be of shape K * K !!!")

		self.vertices=np.zeros((nVertices,self.K))
		self.adjacencyMat=np.zeros((nVertices,nVertices))
		self.trueClusters=np.zeros(self.nVertices, dtype=np.int8)
		self.kmeansClusters=np.zeros(self.nVertices, dtype=np.int8)

	def simulateVertices(self):
		# sample vertex profils
		for i in range(self.nVertices):
			self.vertices[i,:] = np.random.dirichlet(self.alpha)
		# sort vertices by highest value
		self.vertices = np.array(sorted(self.vertices, key=lambda x: np.argmax(x)))# ,1-np.max(x))))

	def simulateMatching(self, Vert1, Vert2):
		z1 = np.random.multinomial(1, Vert1, size=1)
		z1 = z1[0]
		z2 = np.random.multinomial(1, Vert2, size=1)
		z2 = z2[0]
		probaMatching = (1.-self.rho) * z1@(self.bernMat@z2)
		if np.random.rand() < probaMatching:
			return 1
		else:
			return 0

	def simulateEdges(self):
		for i in range(self.nVertices):
			for j in range(i,self.nVertices):
				matching_ij = self.simulateMatching(self.vertices[i,:],self.vertices[j,:])
				self.adjacencyMat[i,j] = matching_ij
				self.adjacencyMat[j,i] = matching_ij

	def generateGraph(self, plot=False):
		self.simulateVertices()
		self.simulateEdges()

	def showGraph(self):
		plt.figure(figsize=(13,6))
		# drawing adjacency matrix
		plt.subplot(131)
		matPlot = np.zeros((self.nVertices, self.nVertices, 3))
		matPlot[self.adjacencyMat==0,:] = np.array([255,255,255])
		matPlot[self.adjacencyMat==1,:] = np.array([0, 153, 255])
		plt.imshow(matPlot)
		plt.title("adjacency matrix")
		# drawing vertices
		plt.subplot(132)
		plt.imshow(self.vertices, cmap='Reds')
		plt.title("vertices profil")
		plt.colorbar()
		# drawing graph
		G = nx.Graph()
		G.add_nodes_from(list(range(self.nVertices)))
		for i in range(self.nVertices):
			for j in range(i,self.nVertices):
				if self.adjacencyMat[i,j] == 1:
					G.add_edge(i,j)
		plt.subplot(133)
		plt.title('graph')
		nx.draw(G, with_labels=True)
		plt.show()

	def groundTruthClustering(self,colorClusters=['pink','cyan','orange','green','purple','brown']):

		color_map = []
		for i in range(self.nVertices):
			self.trueClusters[i] = int(np.argmax(self.vertices[i,:]))
			color_map.append(colorClusters[self.trueClusters[i]])

		# draw graph with ground truth clusters
		G = nx.Graph()
		G.add_nodes_from(list(range(self.nVertices)))
		for i in range(self.nVertices):
			for j in range(i,self.nVertices):
				if self.adjacencyMat[i,j] == 1:
					G.add_edge(i,j)
		#plt.subplot(122)
		plt.title('ground truth clustering')
		nx.draw(G, node_color=color_map, with_labels=True)
		plt.show()

	def kmeansClustering(self,colorClusters=['pink','cyan','orange','green','purple','brown']):

		# infer kmeans clusters
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.adjacencyMat)
		self.kmeansClusters=np.array(self.K-1-kmeans.labels_,dtype=np.int8)

		# matching kmeans clusters labels with ground truth clusters labels
		bestSimil=0
		for perm in list(permutations(list(range(self.K)))): 
			permKClusters=np.zeros(len(self.trueClusters),dtype=np.int8) 
			permKClusters[self.kmeansClusters==0]=perm[0]
			permKClusters[self.kmeansClusters==1]=perm[1]
			permKClusters[self.kmeansClusters==2]=perm[2]
			simil = np.sum(permKClusters==self.trueClusters)
			if simil>bestSimil:
				bestSimil=simil
				matchedKClusters=permKClusters
		self.kmeansClusters = matchedKClusters
		print('\nKmeans label recovery error: {:.1f}%\n'.format(100*(self.nVertices-bestSimil)/self.nVertices))

		km_color_map = []
		for i in range(self.nVertices):
			km_color_map.append(colorClusters[self.kmeansClusters[i]])

		# infer ground truth clusters
		gt_color_map = []
		for i in range(self.nVertices):
			self.trueClusters[i] = int(np.argmax(self.vertices[i,:]))
			gt_color_map.append(colorClusters[self.trueClusters[i]])

		# draw graph with kmeans clusters
		G = nx.Graph()
		G.add_nodes_from(list(range(self.nVertices)))
		for i in range(self.nVertices):
			for j in range(i,self.nVertices):
				if self.adjacencyMat[i,j] == 1:
					G.add_edge(i,j)
		#plt.subplot(133)
		plt.title('Kmeans clustering')

		nx.draw(G, node_color=km_color_map, with_labels=True)

		plt.show()

	def getDegreesMatrix(self):
		diag = []
		for i in range(self.nVertices):
			diag.append(np.sum(self.adjacencyMat[i,:]))
		return np.diag(diag)


	def spectralClustering(self,colorClusters=['pink','cyan','orange','green','purple','brown']):

		Laplacian = self.getDegreesMatrix() - self.adjacencyMat
		Lsym = np.diag(np.diag(self.getDegreesMatrix())**-0.5) @ Laplacian @ np.diag(np.diag(self.getDegreesMatrix())**-0.5)

		ev, U = np.linalg.eig(Lsym)

		plt.figure(figsize=(13,6))
		plt.subplot(131)
		plt.scatter(list(range(self.nVertices)), ev,marker = 'd',c='red')
		plt.title('eigenvalues')

		plt.subplot(132)
		plt.plot(U[:,0],'-o', label='eigenvector 1')
		plt.plot(U[:,1],'-o', label='eigenvector 2')
		plt.plot(U[:,2],'-o', label='eigenvector 3')
		plt.legend(loc='upper left')

		T_norm = np.tile(1./np.sqrt(np.sum(U[:,:self.K]**2, axis=1)),(self.K,1)).transpose() # row normalization
		X_spec_norm = np.multiply(U[:,:self.K], T_norm)

		# repeat same steps as for KMeans
		# infer kmeans clusters
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(X_spec_norm)
		#print(kmeans.labels_)
		#self.kmeansClusters=np.array(self.K-1-kmeans.labels_,dtype=np.int8)

		# matching kmeans clusters labels with ground truth clusters labels
		bestSimil=0
		for perm in list(permutations(list(range(self.K)))): 
			permKClusters=np.zeros(len(self.trueClusters),dtype=np.int8) 
			permKClusters[kmeans.labels_==0]=perm[0]
			permKClusters[kmeans.labels_==1]=perm[1]
			permKClusters[kmeans.labels_==2]=perm[2]
			simil = np.sum(permKClusters==self.trueClusters)
			if simil>bestSimil:
				bestSimil=simil
				matchedKClusters=permKClusters
		self.kmeansClusters = np.array(matchedKClusters)
		print('\nSpectral clustering label recovery error: {:.1f}%\n'.format(100*(self.nVertices-bestSimil)/self.nVertices))

		km_color_map = []
		for i in range(self.nVertices):
			km_color_map.append(colorClusters[self.kmeansClusters[i]])

		# infer ground truth clusters
		gt_color_map = []
		for i in range(self.nVertices):
			self.trueClusters[i] = int(np.argmax(self.vertices[i,:]))
			gt_color_map.append(colorClusters[self.trueClusters[i]])

		# draw graph with kmeans clusters
		plt.subplot(133)
		G = nx.Graph()
		G.add_nodes_from(list(range(self.nVertices)))
		for i in range(self.nVertices):
			for j in range(i,self.nVertices):
				if self.adjacencyMat[i,j] == 1:
					G.add_edge(i,j)
		#plt.subplot(133)
		plt.title('Spectral clustering')

		nx.draw(G, node_color=km_color_map, with_labels=True)

		plt.show()

class CloudGraph:
	def __init__(self, cloudPoints, trueClusters):
		self.cloudPoints = cloudPoints
		self.adjacencyMat = None
		self.trueClusters = trueClusters
		self.K = len(np.unique(self.trueClusters))
		self.nVertices = len(self.cloudPoints)
		self.n_ngb=0
		# rearrange vectors based on labels
		self.cloudPoints=np.vstack( (self.cloudPoints[self.trueClusters==0,:], self.cloudPoints[self.trueClusters==1,:]))
		self.trueClusters=np.hstack( (self.trueClusters[self.trueClusters==0], self.trueClusters[self.trueClusters==1]))

	def generateMutualKNNGraph(self, n_ngb=5, sym='tight'):
		self.n_ngb = n_ngb
		self.adjacencyMat = kneighbors_graph(self.cloudPoints, n_neighbors=n_ngb, include_self=True).toarray()
		# make adjacency matrix symmetrical
		if sym=='loose':
			self.adjacencyMat = np.maximum(self.adjacencyMat,self.adjacencyMat.transpose())
		else:
			self.adjacencyMat = np.multiply(self.adjacencyMat,self.adjacencyMat.transpose())

	def showGraph(self):
		plt.figure(figsize=(12,6))

		# drawing adjacency matrix
		plt.subplot(131)
		print(self.nVertices)
		matPlot = np.zeros((self.nVertices, self.nVertices, 3))
		matPlot[self.adjacencyMat==0] = np.array([255,255,255])
		matPlot[self.adjacencyMat>0] = np.array([255, 0, 255])#([0, 153, 255])
		plt.imshow(matPlot)
		plt.title("adjacency matrix")
		# plotting cloud data points
		plt.subplot(132)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in self.trueClusters ])
		plt.axis('off')
		plt.title('point cloud data')
		# plot edges
		plt.subplot(133)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in self.trueClusters ])
		plt.axis('off')
		plt.title('{}-NN graph on PCD'.format(self.n_ngb))
		for i in range(self.nVertices-1):
			for j in range(i+1,self.nVertices):
				if self.adjacencyMat[i,j]==1:
					plt.plot(self.cloudPoints[[i,j],0], self.cloudPoints[[i,j],1], 'k-', linewidth=0.2)
		plt.show()

	def kmeansClustering(self):
		plt.figure(figsize=(13,6))

		# infer kmeans clusters from PCD
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.cloudPoints)

		# matching kmeans clusters labels with ground truth clusters labels
		bestSimil=0
		for perm in list(permutations(list(range(self.K)))): 
			permKClusters=np.zeros(len(self.trueClusters),dtype=np.int8) 
			permKClusters[kmeans.labels_==0]=perm[0]
			permKClusters[kmeans.labels_==1]=perm[1]
			simil = np.sum(permKClusters==self.trueClusters)
			if simil>bestSimil:
				bestSimil=simil
				matchedKClusters=permKClusters
		kmeansClusters = np.array(matchedKClusters)
		print('\nKmeans on PCD > label recovery error: {:.1f}%\n'.format(100*(self.nVertices-bestSimil)/self.nVertices))

		plt.subplot(121)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in kmeansClusters ])
		plt.axis('off')
		plt.title('Kmeans clustering on PCD')

		# infer kmeans clusters from points in adjacency representation
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.adjacencyMat)

		# matching kmeans clusters labels with ground truth clusters labels
		bestSimil=0
		for perm in list(permutations(list(range(self.K)))): 
			permKClusters=np.zeros(len(self.trueClusters),dtype=np.int8) 
			permKClusters[kmeans.labels_==0]=perm[0]
			permKClusters[kmeans.labels_==1]=perm[1]
			simil = np.sum(permKClusters==self.trueClusters)
			if simil>bestSimil:
				bestSimil=simil
				matchedKClusters=permKClusters
		kmeansClusters = np.array(matchedKClusters)
		print('\nKmeans on adjacency matrix > label recovery error: {:.1f}%\n'.format(100*(self.nVertices-bestSimil)/self.nVertices))

		plt.subplot(122)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in kmeansClusters ])
		plt.axis('off')
		plt.title('Kmeans clustering on adjacency matrix')

		plt.show()

	def getDegreesMatrix(self):
		diag = []
		for i in range(self.nVertices):
			diag.append(np.sum(self.adjacencyMat[i,:]))
		return np.diag(diag)

	def spectralClustering(self):

		Laplacian = self.getDegreesMatrix() - self.adjacencyMat
		Lsym = np.diag(np.diag(self.getDegreesMatrix())**-0.5) @ Laplacian @ np.diag(np.diag(self.getDegreesMatrix())**-0.5)

		ev, U = np.linalg.eig(Lsym)

		plt.figure(figsize=(13,6))
		plt.subplot(131)
		plt.scatter(list(range(self.nVertices)), ev, marker = 'd',c='red')
		plt.title('eigenvalues')

		plt.subplot(132)
		plt.plot(U[:,0],'-o', label='eigenvector 1')
		plt.plot(U[:,1],'-o', label='eigenvector 2')
		plt.legend(loc='upper right')

		T_norm = np.tile(1./np.sqrt(np.sum(U[:,:self.K]**2, axis=1)),(self.K,1)).transpose() # row normalization
		X_spec_norm = np.multiply(U[:,:self.K], T_norm)

		# repeat same steps as for KMeans
		# infer kmeans clusters
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(X_spec_norm)
		#self.kmeansClusters=np.array(self.K-1-kmeans.labels_,dtype=np.int8)

		# matching kmeans clusters labels with ground truth clusters labels
		bestSimil=0
		for perm in list(permutations(list(range(self.K)))): 
			permKClusters=np.zeros(len(self.trueClusters),dtype=np.int8) 
			permKClusters[kmeans.labels_==0]=perm[0]
			permKClusters[kmeans.labels_==1]=perm[1]
			simil = np.sum(permKClusters==self.trueClusters)
			if simil>bestSimil:
				bestSimil=simil
				matchedKClusters=permKClusters
		kmeansClusters = np.array(matchedKClusters)
		print('\nSpectral clustering label recovery error: {:.1f}%\n'.format(100*(self.nVertices-bestSimil)/self.nVertices))

		plt.subplot(133)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in kmeansClusters ])
		plt.axis('off')
		plt.title('Spectral clustering')

		plt.show()