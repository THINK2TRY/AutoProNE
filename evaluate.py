import argparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from collections import defaultdict
from scipy import sparse
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

from utils import load_labels, load_embedding


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sparse.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels


def classifier(X_train, y_train, X_test, y_test):
    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train)

    # find out how many labels should be predicted
    top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
    preds = clf.predict(X_test, top_k_list)

    results = {}
    averages = ["micro", "macro", "samples", "weighted"]
    for average in averages:
        results[average] = f1_score(y_test, preds, average=average)
    return results


def evaluate(emb, number_shuffles=5, label=None):
    if type(emb) == str:
        features_matrix = load_embedding(args.emb)
    else:
        features_matrix = emb
    print(features_matrix.shape)
    num_nodes = features_matrix.shape[0]

    if label is None:
        label_matrix = load_labels(args.label, num_nodes)
    else:
        label_matrix = load_labels(label, num_nodes)

    shuffles = []
    for x in range(number_shuffles):
        shuffles.append(skshuffle(features_matrix, label_matrix, random_state=0))

    all_results = defaultdict(list)

    if num_nodes < 20000:
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]
    else:
        training_percents = [0.01, 0.03, 0.05, 0.07, 0.09]

    for train_percent in training_percents:
        # pool = multiprocessing.Pool(processes=shuffles)
        # results = []
        for shuf in shuffles:
            X, y = shuf
            training_size = int(train_percent * num_nodes)

            X_train = X[:training_size, :]
            y_train = y[:training_size, :]

            X_test = X[training_size:, :]
            y_test = y[training_size:, :]

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro", "samples", "weighted"]
            for average in averages:
                results[average] = f1_score(y_test, preds, average=average)
            all_results[train_percent].append(results)

        #     results.append(pool.apply_async(classifier, (X_train, y_train, X_test, y_test)))
        # pool.close()
        # pool.join()
        # for res in results:
        #     all_results[train_percent].append(res.get())

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
    parser.add_argument('--label', nargs='?', default='data/PPI.cmty',
                        help='Input label file path')
    parser.add_argument('--emb', nargs='?', default='emb/PPI.emb',
                        help='embeddings file path')
    parser.add_argument('--shuffle', type=int, default=4,
                        help='number of shuffle')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.emb, args.shuffle, args.label)
