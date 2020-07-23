from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import math

from filter_module import NodeAdaptiveEncoder
from spectral_prop import get_embedding_dense
from spectral_prop import propagate
from utils import load_embedding, load_adjacency_mx, sigmoid


class ContraSearch(object):
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
        options = []
        for pt in self.prop_types:
            if "heat" == pt:
                options.append({
                    "type": "heat",
                    "t": hp.uniform("t", 0.1, 0.9)
                })
            if "ppr" == pt:
                options.append({
                    "type": "ppr",
                    "alpha": hp.uniform("alpha", 0.1, 0.9)
                })
            if "gaussian" == pt:
                options.append({
                    "type": "gaussian",
                    "theta": hp.uniform("theta", 0.5, 3),
                    "mu": hp.uniform("mu", -4, 2),
                    # "rescale": hp.choice("rescale", [0, 1])
                })
            if "sc" == pt:
                options.append({
                    "type": "sc"
                })
        space = hp.choice('filter_type', options)
        return space

    def target_func(self, params):
        prop_result_emb, neg_prop_result_emb = self.prop(params)
        # return {"loss": self.loss_func(prop_result_emb), "status": STATUS_OK}
        return {"loss": self.info_loss(prop_result_emb, neg_prop_result_emb), "status": STATUS_OK}

    def prop(self, params):
        selected_prop = params["type"]
        print(selected_prop, params)
        prop_result = propagate(self.adj, self.emb, selected_prop, params)

        # --
        def get_permute():
            return np.random.permutation(np.arange(self.num_nodes))
        # neg_prop_result = []
        # for s_prop in self.prop_types:
        #     neg_prop = propagate(self.adj, self.emb[get_permute()], s_prop, params)
        #     neg_prop_result.append(neg_prop)
        neg_prop_result = propagate(self.adj, self.emb[get_permute()], selected_prop, params)

        prop_result = get_embedding_dense(prop_result, prop_result.shape[1])
        neg_prop_result = get_embedding_dense(neg_prop_result, neg_prop_result.shape[1])
        return prop_result, neg_prop_result

    def loss_func(self, prop_emb):
        # Sparest Cut Loss
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1])**2, dim=1).mean()
        return 1./loss

    def info_loss(self, prop_emb, neg_prop_emb_list):
        pos_glb = np.mean(prop_emb, axis=0)
        # neg_glb = np.sum(neg_prop_emb, axis=0)

        pos_info = pos_glb.dot(prop_emb.T)
        pos_loss = np.mean(np.log(sigmoid(pos_info)))

        # neg_loss = 0
        # for neg_prop_emb in neg_prop_emb_list:
        #     neg_info = pos_glb.dot(neg_prop_emb.T)
        #     neg_loss += np.mean(np.log(1 - sigmoid(neg_info)))

        neg_info = pos_glb.dot(neg_prop_emb_list.T)
        neg_loss = np.mean(np.log(1 - sigmoid(neg_info)))

        semi_pos_info = pos_glb.dot(self.emb.T)
        semi_loss = np.mean(np.log(1 - sigmoid(semi_pos_info)))

        return -(pos_loss + neg_loss + semi_loss) / 3
        # return -(pos_loss + neg_loss + semi_loss) / (2 + len(neg_prop_emb_list))

    def __call__(self):
        search_space = self.build_search_space()

        trials = Trials()
        best = fmin(self.target_func, search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        best["type"] = self.prop_types[best["filter_type"]]

        print(f"best parameters: {best}")

        with open("search_.log", "w") as f:
            for trial in trials:
                f.write(str(trial) + "\n")

        best_result = self.prop(best)[0]

        return best_result
