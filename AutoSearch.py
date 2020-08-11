from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import math

from filter_module import NodeAdaptiveEncoder
from spectral_prop import propagate, get_embedding_dense
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
        if "heat" in self.prop_types:
            space["t"] = hp.uniform("t", 0.2, 0.8)
        if "gaussian" in self.prop_types:
            space["mu"] = hp.uniform("mu", -4, 2)
            space["theta"] = hp.uniform("theta", 0, 4)
            space["rescale"] = hp.choice("rescale", [0, 1])
        if "ppr" in self.prop_types:
            space["alpha"] = hp.uniform("alpha", 0.2, 0.8)
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
            prop_result.append(propagate(self.adj, self.emb, tp, params))
        neg_prop_result = []

        def get_permute():
            return np.random.permutation(np.arange(self.num_nodes))

        for idx, tp in enumerate(self.prop_types):
            neg_prop_result.append(propagate(self.adj, self.emb[get_permute()], tp, params))
        return prop_result, neg_prop_result, weights

    def loss_func(self, prop_emb):
        # Sparest Cut Loss
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1])**2) / self.negative_pairs * self.num_edges
        return 1./loss

    def info_loss(self, prop_emb_list, neg_emb_list, weight_list):
        prop_emb = sum([prop_emb_list[i] * weight_list[i] for i in range(len(weight_list))])
        pos_glb = prop_emb.mean(0)
        pos_info = pos_glb.dot(prop_emb.T)
        pos_loss = np.mean(np.log(sigmoid(pos_info)))

        neg_loss = 0
        for i, neg in enumerate(neg_emb_list):
            neg_info = pos_glb.dot(neg.T)
            neg_loss += np.mean(np.log(1 - sigmoid(neg_info))) * weight_list[i]

        semi_info = prop_emb.dot(self.emb.T)
        semi_loss = np.mean(np.log(1 - sigmoid(semi_info)))

        return -(pos_loss + neg_loss + semi_loss) / 3

    def __call__(self):
        search_space = self.build_search_space()

        trials = Trials()
        best = fmin(self.target_func, search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        print(f"best parameters: {best}")

        with open("search_.log", "w") as f:
            for trial in trials:
                f.write(str(trial) + "\n")

        best_result, _, weights = self.prop(best)

        return sum(x * y for x, y in zip(best_result, weights))


class PlainFilter(object):
    def __init__(self, args):
        self.prop_types = args.prop_types
        self.dim = args.dim
        self.svd = args.svd
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
        prop_result_emb = np.concatenate(prop_result, axis=1)
        if self.svd:
            prop_result_emb = get_embedding_dense(prop_result_emb, self.dim)
        return prop_result_emb
