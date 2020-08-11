import numpy as np

from spectral_propagation import propagate, get_embedding_dense
from utils import load_embedding, load_adjacency_mx


class PlainFilter(object):
    def __init__(self, args):
        self.prop_types = args.prop_types
        self.svd = args.svd
        # load adjacency matrix and raw embedding
        self.emb = load_embedding(args.emb)
        self.adj, self.num_nodes, self.num_edges = load_adjacency_mx(args.adj)
        self.dim = self.emb.shape[1]

    def __call__(self):
        if len(self.prop_types) == 1 and self.prop_types[0] == "identity":
            return self.emb

        prop_result = []
        for tp in self.prop_types:
            prop_result.append(propagate(self.adj, self.emb, tp))
        prop_result_emb = np.concatenate(prop_result, axis=1)
        if self.svd:
            prop_result_emb = get_embedding_dense(prop_result_emb, self.dim)
        return prop_result_emb
