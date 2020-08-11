from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import math

from filter_module import NodeAdaptiveEncoder
from spectral_prop import propagate
from utils import load_embedding, load_adjacency_mx, sigmoid


class SearchFilter(object):
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
        # space["node_adapt"] = hp.choice("node_adapt", [0, 1])
        space["prop_type"] = hp.choice("prop_type", self.prop_types)
        if "heat" in self.prop_types:
            space["t"] = hp.uniform("t", 0.1, 0.9)
        if "gaussian" in self.prop_types:
            space["mu"] = hp.uniform("mu", -4, 2)
            space["theta"] = hp.uniform("theta", 0.5, 3)
            space["rescale"] = hp.choice("rescale", [0, 1])
        if "ppr" in self.prop_types:
            space["alpha"] = hp.uniform("alpha", 0.1, 0.9)
        return space

    def target_func(self, params):
        prop_result_emb, neg_prop_result_emb = self.prop(params)
        # return {"loss": self.loss_func(prop_result_emb), "status": STATUS_OK}
        return {"loss": self.info_loss(prop_result_emb, neg_prop_result_emb), "status": STATUS_OK}

    def prop(self, params):
        selected_prop = params["prop_type"]
        prop_result = propagate(self.adj, self.emb, selected_prop, params)

        # --
        neg_order = np.random.permutation(np.arange(self.num_nodes))
        neg_prop_result = propagate(self.adj, self.emb[neg_order], selected_prop, params)
        return prop_result, neg_prop_result

    def loss_func(self, prop_emb):
        # Sparest Cut Loss
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1])**2, dim=1).mean()
        return 1./loss

    def info_loss(self, prop_emb, neg_prop_emb):
        pos_glb = np.mean(prop_emb, axis=0)
        # neg_glb = np.sum(neg_prop_emb, axis=0)

        pos_info = pos_glb.dot(prop_emb.T)
        neg_info = pos_glb.dot(neg_prop_emb.T)
        pos_loss = np.mean(np.log(sigmoid(pos_info)))
        neg_loss = np.mean(np.log(1 - sigmoid(neg_info)))

        semi_pos_info = pos_glb.dot(self.emb.T)
        semi_loss = np.mean(np.log(1 - sigmoid(semi_pos_info))) / 3

        return -(pos_loss + neg_loss) / 3 + semi_loss

    def __call__(self):
        search_space = self.build_search_space()

        trials = Trials()
        best = fmin(self.target_func, search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)

        # with open("search_.log", "w") as f:
        #     for trial in trials:
        #         f.write(str(trial) + "\n")

        best["prop_type"] = self.prop_types[best["prop_type"]]
        best_result = self.prop(best)[0]
        print(f"best parameters: {best}")

        return best_result
