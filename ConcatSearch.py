from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import math

from filter_module import NodeAdaptiveEncoder
from spectral_prop import get_embedding_dense
from spectral_prop import propagate
from utils import load_embedding, load_adjacency_mx, sigmoid


class ConcatSearch(object):
    def __init__(self, args):
        self.prop_types = args.prop_types
        self.max_evals = args.max_evals

        # load adjacency matrix and raw embedding
        self.emb = load_embedding(args.emb)
        self.adj, self.num_nodes, self.num_edges = load_adjacency_mx(args.adj)

        assert self.num_nodes == self.emb.shape[0]
        # negative pairs
        self.negative_pairs = int(math.sqrt(self.num_edges))

    def build_search_space(self):
        space = {}
        for f in self.prop_types:
            space[f] = hp.choice(f, [0, 1])
        if "heat" in self.prop_types:
            space["t"] = hp.uniform("t", 0.1, 0.9)
        if "gaussian" in self.prop_types:
            space["mu"] = hp.uniform("mu", -4, 2)
            space["theta"] = hp.uniform("theta", 0.5, 3)
        if "ppr" in self.prop_types:
            space["alpha"] = hp.uniform("alpha", 0.1, 0.9)
        return space

    def target_func(self, params):
        prop_result_emb, neg_prop_result_emb = self.prop(params)
        # return {"loss": self.loss_func(prop_result_emb), "status": STATUS_OK}
        return {"loss": self.info_loss(prop_result_emb, neg_prop_result_emb), "status": STATUS_OK}

    def remove_param(self, prop_list, param):
        for j in self.prop_types:
            if j not in prop_list:
                if j == "heat":
                    del param["t"]
                if j == "ppr":
                    del param["alpha"]
                if j == "gaussian":
                    del param["mu"]
                    del param["theta"]
        return param

    def prop(self, params):
        prop_types = [key for key, value in params.items() if value == 1]
        if not prop_types:
            prop_types = self.prop_types
        params = self.remove_param(prop_types, params)
        print(prop_types, params)

        prop_result_list = []
        for selected_prop in prop_types:
            prop_result = propagate(self.adj, self.emb, selected_prop, params)
            prop_result_list.append(prop_result)

        def get_permute():
            return np.random.permutation(np.arange(self.num_nodes))

        neg_prop_result = []
        pmt = get_permute()
        for s_prop in prop_types:
            neg_prop = propagate(self.adj, self.emb[pmt], s_prop, params)
            neg_prop[pmt] = neg_prop
            neg_prop_result.append(neg_prop)

        return prop_result_list, neg_prop_result

    def loss_func(self, prop_emb):
        # Sparest Cut Loss
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1]) ** 2, dim=1).mean()
        return 1. / loss

    def info_loss(self, prop_emb_list, neg_prop_emb_list):
        prop_result = np.concatenate(prop_emb_list, axis=1)
        prop_result = get_embedding_dense(prop_result, prop_result.shape[1])
        pos_glb = prop_result.mean(0)

        neg_prop_result = np.concatenate(neg_prop_emb_list, axis=1)
        neg_prop_result = get_embedding_dense(neg_prop_result, neg_prop_result.shape[1])

        pos_info = sigmoid(pos_glb.dot(prop_result.T))
        neg_info = sigmoid(pos_glb.dot(neg_prop_result.T))
        pos_loss = np.mean(np.log(pos_info)).mean()
        neg_loss = np.mean(np.log(1 - neg_info)).mean()

        return -(pos_loss + neg_loss) / 2

    def __call__(self):
        search_space = self.build_search_space()
        trials = Trials()
        best = fmin(self.target_func, search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)

        with open("search_.log", "w") as f:
            for trial in trials:
                f.write(str(trial) + "\n")

        best_result = self.prop(best)[0]
        best_result = np.concatenate(best_result, axis=1)
        print(f"best parameters: {best}")

        best_result = get_embedding_dense(best_result, best_result.shape[1])
        return best_result
