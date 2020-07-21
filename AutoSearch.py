from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import math

from filter_module import NodeAdaptiveEncoder
from spectral_prop import propagate
from utils import load_embedding, load_adjacency_mx, sigmoid


class AutoSearch(object):
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
        for prop_type in self.prop_types:
            space[prop_type] = hp.uniform(prop_type, 0, 1)
        space["node_adapt"] = hp.choice("node_adapt", [0, 1])
        space["k"] = hp.choice("k", [1, 2, 3, 4, 5])
        if "heat" in self.prop_types:
            space["t"] = hp.uniform("t", 0, 1)
        if "gaussian" in self.prop_types:
            space["mu"] = hp.uniform("mu", -4, 2)
            space["theta"] = hp.uniform("theta", 0, 4)
            space["rescale"] = hp.choice("rescale", [0, 1])
        if "ppr" in self.prop_types:
            space["alpha"] = hp.uniform("alpha", 0, 1)
        return space

    def target_func(self, params):
        prop_result_emb, prop_result_list, weights = self.prop(params)
        # return {"loss": self.loss_func(prop_result_emb), "status": STATUS_OK}
        return {"loss": self.info_loss(prop_result_emb, prop_result_list, weights), "status": STATUS_OK}

    def prop(self, params):
        weights = [params[x] for x in self.prop_types]
        assert sum(weights) > 0
        weights = [x/sum(weights) for x in weights]
        prop_result = []
        for idx, tp in enumerate(self.prop_types):
            prop_result.append(propagate(self.adj, self.emb, tp, params) * weights[idx])
        prop_result_emb = sum(prop_result)
        if params["node_adapt"] == 1:
            prop_result_emb = NodeAdaptiveEncoder.prop(prop_result_emb)
        return prop_result_emb, prop_result, weights

    def loss_func(self, prop_emb):
        # Sparest Cut Loss
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1])**2) / self.negative_pairs * self.num_edges
        return 1./loss

    def info_loss(self, prop_emb, prop_emb_list, weight_list):
        pos_emb = sum([weight_list[i] * prop_emb * prop_emb_list[i] for i in range(len(prop_emb_list))])
        for i in range(len(prop_emb_list)):
            np.random.shuffle(prop_emb_list[i])
        neg_emb = sum([weight_list[i] * prop_emb * prop_emb_list[i] for i in range(len(prop_emb_list))])
        pos_loss = np.sum(np.log(sigmoid(np.sum(pos_emb, axis=1)))) / self.num_nodes
        neg_loss = np.sum(np.log(1 - sigmoid(np.sum(neg_emb, axis=1)))) / self.num_nodes
        return -(pos_loss + neg_loss) / 2

    def __call__(self):
        search_space = self.build_search_space()

        trials = Trials()
        best = fmin(self.target_func, search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        print(f"best parameters: {best}")

        with open("search_.log", "w") as f:
            for trial in trials:
                f.write(str(trial))

        best["k"] = best["k"] + 1
        best_result = self.prop(best)

        return best_result[0]


class PlainFilter(object):
    def __init__(self, args):
        self.prop_types = args.prop_types

        # load adjacency matrix and raw embedding
        self.emb = load_embedding(args.emb)
        self.adj, self.num_nodes, self.num_edges = load_adjacency_mx(args.adj)
        self.rescale = args.rescale

    def __call__(self):
        # propagate the embedding
        if len(self.prop_types) == 1 and self.prop_types[0] == "identity":
            return self.emb

        prop_result = []
        for tp in self.prop_types:
            prop_result.append(propagate(self.adj, self.emb, tp, resale=self.rescale))
        prop_result_emb = sum(prop_result) / len(prop_result)
        # prop_result_emb = NodeAdaptiveEncoder.prop(prop_result_emb)

        return prop_result_emb
