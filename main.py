'''
This file contains the main code for the project
1. Read 'train.csv' and 'test.csv' files 
2. Preprocess the files
3. Perform LSH on the files
- Finds 2000 similar rows for a query vector 
4. Perform K Nearest Neighbours on the LSH result 
- Finds 1000 similar neighbours 
5. Pipeline of classification learning algorithms: Neural Network -> SVM 
'''

import pandas as pd 
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import svm

import preprocess
import lsh

class project():
	
	def __init__(self):
		self.train_dataframe = pd.read_csv('data/training.csv', header=0) 
		self.test_dataframe = pd.read_csv('data/test.csv', header=0)
		self.test_dataframe_refId = self.test_dataframe['RefId']
		
		self.preprocess_data()
		
		# Initialise classifiers
		self.attributes = list(self.train_dataframe.columns.values)[1:]					
		self.lsh_neighbours = 2000
		self.initialise_knn()
		self.initialise_pca()
		self.initialise_svm()
		self.initialise_nn()
		
		self.lsh = lsh.lsh(self.train_dataframe)
		
	def preprocess_data(self):
		print "Preprocessing Data"
		self.train_dataframe = preprocess.preprocess(self.train_dataframe)
		self.test_dataframe = preprocess.preprocess(self.test_dataframe)
		
		# Add dummy column to test dataframe to match dimensions
		# Quick hack: should take away
		self.test_dataframe['IsBadBuy'] = 0
				
	def initialise_knn(self):
		print "Initialising KNN" 
		k = np.sqrt(self.lsh_neighbours/2)
		#k = 150	# Testing
		self.knn_clf = KNeighborsClassifier(n_neighbors=k)
		
	def initialise_pca(self):
		print "Initialsing PCA"
		self.pca_clf = PCA(n_components=len(self.attributes)/2)
		
	def initialise_svm(self):
		print "Initialising SVM"
		self.svm_clf = svm.SVC(kernel='linear')
		
	def initialise_nn(self):
		print "Initialising Neural Network"
		num_hidden_nodes = 3
		learning_rate = 0.05
		batch_size = 30
		self.nn_clf = BernoulliRBM(n_components=num_hidden_nodes, learning_rate=learning_rate, batch_size=batch_size)
		
	def run(self):
		predictions = []
		refId = []
		
		for idx, row in self.test_dataframe.iterrows():
			print "Querying LSH"
			# query_vector = self.train_dataframe.iloc[1]	# Testing query vector
			query_vector = row
			
			lsh_idx = self.lsh.query(query_vector, self.lsh_neighbours)
			#print lsh_idx
			
			print "K Nearest Neighbours"	
			kneighbours = self.k_nearest_neighbours(lsh_idx, query_vector)
			
			# For PCA
			#train_pca, query_pca = self.perform_pca(kneighbours, query_vector)
			#prediction = self.neural_network(train_pca, query_pca)
			
			try: 
				prediction = self.neural_network(self.train_dataframe.ix[kneighbours], query_vector)
			except:
				prediction = 0
			predictions.append(prediction)
			refId.append(self.test_dataframe_refId.ix[idx])
			
			# print str(prediction) + " " + str(self.test_dataframe_refId.ix[idx])
			
			# Quick hack for testing
			'''
			if idx == 3:
				break
			'''
			
		self.output_data(predictions, refId)
		
	def k_nearest_neighbours(self, lsh_idx, query_vector): 
		'''
		This function finds num_neighbours k-nearest-neighbours
		- Default k value: sqrt(num_k_neighbours/2)
		- Default Distance: Euclidean
		Reference: http://blog.yhathq.com/posts/classification-using-knn-and-python.html
		
		Returns: np.array([]) of row indices of dataframe that are closest to query vector
		TODO: Graph of accuracy as k increases? Or modify how to calculate distance between points
		'''
		lsh_dataframe = self.train_dataframe.ix[lsh_idx]
		self.knn_clf.fit(lsh_dataframe[self.attributes], lsh_dataframe['IsBadBuy'])		
		neighbours = self.knn_clf.kneighbors(query_vector[self.attributes], return_distance=False)
		
		# print neighbours
		return neighbours.flatten()
		
	def perform_pca(self, kneighbours, query_vector):
		print "Performing PCA"
		dataframe = self.train_dataframe.ix[kneighbours]
		self.pca_clf.fit(dataframe)
		components = self.pca_clf.components_ 
		
		train_pca = self.pca_clf.transform(dataframe)
		query_pca = self.pca_clf.transform(query_vector)
		
		return train_pca.flatten(), query_pca.flatten()
		
	def neural_network(self, dataframe, query_vector):
		'''
		This function trains a neural network based on a PCA transformed dataframe and query vector
		Using: BernoulliRBM, SVM (because 2 classes) pipeline 
		
		Output: prediction for query vector
		'''
		
		# Drop the predicted variable which was previously put in as dummy to match indices
		query_vector = query_vector.drop(['IsBadBuy']) 
		
		classifier = Pipeline(steps=[('neural', self.nn_clf), ('svm', self.svm_clf)])
		classifier.fit(dataframe[self.attributes], dataframe['IsBadBuy'])
		prediction = classifier.predict(query_vector)
		
		#print prediction
		return prediction[0]
		
	def output_data(self, predictions, refID):
		print "Writing to file"
		array = np.vstack((refID, predictions))
		array_transpose = np.array(np.matrix(array).transpose())
		
		df_results = pd.DataFrame({'RefId': array_transpose[:,0], 'Predicted': array_transpose[:,1]})
		df_results.to_csv('results.csv', index=False, cols=['RefId','Predicted'])
		
if __name__ == "__main__":
	project = project()
	project.run()