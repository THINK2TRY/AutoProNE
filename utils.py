import networkx as nx
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import scipy.sparse as sp
from sklearn import preprocessing


def load_w2v_embeddings(embeddings_file):
    # load embeddings from word2vec format file
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    features_matrix = np.asarray([model[str(node)] for node in range(len(model.index2word))])
    return features_matrix


def save_embedding(emb_file, features):
    # save node embedding into emb_file with word2vec format
    f_emb = open(emb_file, 'w')
    f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
    for i in range(len(features)):
        s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
        f_emb.write(s + "\n")
    f_emb.close()


def load_npy_embedding(path):
    return np.load(path)


def save_npy_embedding(path, data):
    np.save(path, data)


def load_embedding(path):
    if path.endswith("npy"):
        emb = load_npy_embedding(path)
    else:
        emb = load_w2v_embeddings(path)
    return emb


def load_adjacency_mx(adj_path):
    g = nx.read_edgelist(adj_path, nodetype=int, create_using=nx.DiGraph())
    g = g.to_undirected()
    num_nodes = g.number_of_nodes()
    mx = sp.lil_matrix((num_nodes, num_nodes))
    for e in g.edges:
        mx[e[0], e[1]] = 1
        mx[e[1], e[0]] = 1
    return mx.tocsc(), g.number_of_nodes(), g.number_of_edges()


def load_labels(labels_file, nodesize):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('class number: ', len(context))
        label = sp.lil_matrix((nodesize, len(context)))

        for i, line in enumerate(context):
            line = map(int,line.strip().split('\t'))
            for node in line:
                label[node, i] = 1
    return label


def sigmoid(x):
    return 1./(1+np.exp(-x))

# ==============
# just for test
# ==============


def load_random_vector(size, dim):
    return np.random.randn(size, dim)


def generate_adj_mx(m, n, non_zero):
    row = np.random.choice(m, non_zero)
    col = np.random.choice(n, non_zero)
    data = np.ones(non_zero)
    mx = sp.csc_matrix((data, (row, col)), shape=(m, n))
    identity = sp.csc_matrix((np.ones(m), (np.arange(m), np.arange(n))), shape=(m, n))
    mx = mx + identity
    mx = preprocessing.normalize(mx, "l1")
    return mx

# ==============
