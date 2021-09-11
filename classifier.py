import argparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")


class TopKRanker(OneVsRestClassifier):
	def predict(self, X, top_k_list):
		assert X.shape[0] == len(top_k_list)
		probs = np.asarray(super(TopKRanker, self).predict_proba(X))

		# indices = []
		# for i, k in enumerate(top_k_list):
		# 	probs_ = probs[i, :]
		# 	labels = self.classes_[probs_.argsort()[-k:]].tolist()
		# 	indices.extend([[i, x] for x in labels])
		# indices = np.array(indices)
		# indices = indices.transpose()
		# all_labels = sparse.csr_matrix((np.ones(indices.shape[1]), (indices[0], indices[1])), shape=probs.shape)
		# all_labels = all_labels.tolil()

		all_labels = sparse.lil_matrix(probs.shape)
		
		for i, k in enumerate(top_k_list):
			probs_ = probs[i, :]
			labels = self.classes_[probs_.argsort()[-k:]].tolist()
			for label in labels:
				all_labels[i,label] = 1
		return all_labels


def load_embeddings(embeddings_file):
	# load embeddings from word2vec format file
	model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
	features_matrix = np.asarray([model[str(node)] for node in range(len(model.index2word))])
	return features_matrix


def load_labels(labels_file, nodesize):
	# load label from label file, which each line i contains all node who have label i
	with open(labels_file) as f:
		context = f.readlines()
		print('class number: ', len(context))
		label = sparse.lil_matrix((nodesize, len(context)))

		for i, line in enumerate(context):
			line = map(int,line.strip().split('\t'))
			for node in line:
				label[node, i] = 1
	return label


def load_labels_youtube(labels_file, nodesize):
    labeled_nodes = set()
    with open(labels_file) as f:
        context = f.readlines()
        print('class number: ', len(context))
        label = sparse.lil_matrix((nodesize, len(context)))

        for i, line in enumerate(context):
            line = map(int, line.strip().split('\t'))
            for node in line:
                label[node-1, i] = 1
                labeled_nodes.add(node-1)
    labeled_nodes = sorted(list(labeled_nodes))
    labeled_nodes = np.array(labeled_nodes)
    return label.todense()[labeled_nodes], labeled_nodes


def evaluate():
	args = parse_args()
	features_matrix = load_embeddings(args.emb)
	print(features_matrix.shape)
	nodesize = features_matrix.shape[0]

	if "youtube" in args.label:
		label_matrix, labeled_nodes = load_labels_youtube(args.label, nodesize)
		features_matrix = features_matrix[labeled_nodes]
		nodesize = len(labeled_nodes)

	else:
		label_matrix = load_labels(args.label, nodesize)
	number_shuffles = args.shuffle
	
	shuffles = []
	for x in range(number_shuffles):
  		shuffles.append(skshuffle(features_matrix, label_matrix))

	all_results = defaultdict(list)

	if "dblp" or "youtube" in args.emb:
		training_percents = [0.01, 0.03, 0.05, 0.07, 0.09]
	else:
		training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]




	for train_percent in training_percents:
		for shuf in shuffles:
			X, y = shuf
			training_size = int(train_percent * nodesize)

			X_train = X[:training_size, :]
			y_train = y[:training_size, :]

			X_test = X[training_size:, :]
			y_test = y[training_size:,:]

			clf = TopKRanker(LogisticRegression())
			clf.fit(X_train, y_train)

			# find out how many labels should be predicted
			top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
			preds = clf.predict(X_test, top_k_list)

			results = {}
			averages = ["micro", "macro", "samples", "weighted"]
			for average in averages:
				results[average] = f1_score(y_test,  preds, average=average)

			all_results[train_percent].append(results)
	print('Results, using embeddings of dimensionality', X.shape[1])
	print('-------------------')
	print('Train percent:', 'average f1-score')
	for train_percent in sorted(all_results.keys()):
		av = 0
		stder = np.ones(number_shuffles)
		i = 0
		for x in all_results[train_percent]:
			stder[i] = x["micro"]
			i += 1
			av += x["micro"]
		av /= number_shuffles
		print(train_percent, ":", av)


def parse_args():
	parser = argparse.ArgumentParser(description="Community Discover.")
	parser.add_argument('-label', nargs='?', default='data/PPI.cmty',
	                    help='Input label file path')
	parser.add_argument('-emb', nargs='?', default='emb/PPI.emb',
	                    help='embeddings file path')
	parser.add_argument('-shuffle', type=int, default=10,
	                    help='number of shuffule')
	return parser.parse_args()


if __name__ == '__main__':
	evaluate()