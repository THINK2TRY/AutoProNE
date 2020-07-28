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
        self.dim = args.dim
        self.svd = args.svd
        self.loss_type = args.loss

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
            space["mu"] = hp.uniform("mu", 0, 2)
            space["theta"] = hp.uniform("theta", 0.2, 2)
        if "ppr" in self.prop_types:
            space["alpha"] = hp.uniform("alpha", 0.2, 0.8)
        return space

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
                    # del param["rescale"]
        return param

    def prop(self, params):
        prop_types = [key for key, value in params.items() if value == 1 and key in self.prop_types]
        if not prop_types:
            print(" -- dropped -- ")
            prop_types = self.prop_types
            return None, None

        params = self.remove_param(prop_types, params)
        # print(prop_types, params)
        # print("\n")

        prop_result_list = []
        for selected_prop in prop_types:
            prop_result = propagate(self.adj, self.emb, selected_prop, params)
            prop_result_list.append(prop_result)

        def get_permute():
            return np.random.permutation(np.arange(self.num_nodes))

        if self.loss_type == "infomax":
            neg_prop_result = []
            pmt = get_permute()
            for s_prop in prop_types:
                neg_prop = propagate(self.adj, self.emb[pmt], s_prop, params)
                neg_prop[pmt] = neg_prop
                neg_prop_result.append(neg_prop)

            return np.array(prop_result_list), np.array(neg_prop_result)
        elif self.loss_type == "infonce":
            return np.array(prop_result_list), None
        elif self.loss_type == "sparse":
            return np.array(prop_result_list), None
        else:
            raise ValueError("use 'infonce', 'infomax' or 'sparse' loss, currently using {}".format(self.loss_type))

    def target_func(self, params):
        prop_result_emb, neg_prop_result_emb = self.prop(params)
        if prop_result_emb is None:
            return {"loss": 100, "status": STATUS_OK}
        # return {"loss": self.loss_func(prop_result_emb), "status": STATUS_OK}
        if self.loss_type == "infomax":
            loss = self.info_loss(prop_result_emb, neg_prop_result_emb)
        elif self.loss_type == "infonce":
            loss = self.infonce_loss(prop_result_emb)
        else:
            loss = self.sparse_cut_loss(prop_result_emb)
        return {"loss": loss, "status": STATUS_OK}

    def sparse_cut_loss(self, prop_emb_list):
        # Sparest Cut Loss
        prop_emb = np.concatenate(prop_emb_list, axis=1)
        if self.svd:
            prop_emb = get_embedding_dense(prop_emb, self.dim)
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1]) ** 2, axis=1).mean()
        return 1. / loss

    def infonce_loss(self, prop_emb_list, *args, **kwargs):
        batch_size = 64
        T = 0.07
        neg_index = np.random.choice(self.num_nodes, (self.num_nodes, batch_size))
        neg_emb = self.emb[neg_index]

        pos_infos = []
        for smoothed in prop_emb_list:
            pos_info = np.exp(np.sum(smoothed * self.emb, -1) / T)
            assert pos_info.shape == (self.num_nodes,)
            pos_infos.append(pos_info)

        # for i in range(len(prop_emb_list)):
        #     for j in range(i+1, len(prop_emb_list)):
        #         pos_info = np.exp(np.sum(prop_emb_list[i] * prop_emb_list[j], -1) / T)
        #         assert pos_info.shape == (self.num_nodes,)
        #         pos_infos.append(pos_info)

        neg_infos = []
        for idx, smoothed in enumerate(prop_emb_list):
            neg_info = np.exp(np.sum(np.tile(smoothed[:, np.newaxis, :], (1, batch_size, 1)) * neg_emb, -1) / T).sum(-1)
            assert neg_info.shape == (self.num_nodes,)
            neg_infos.append(neg_info + pos_infos[idx])
        # print(np.array(pos_infos) / np.array(neg_infos))
        # input()
        loss = -np.log(np.array(pos_infos) / np.array(neg_infos)).mean()
        return loss/10

    def info_loss(self, prop_emb_list, neg_prop_emb_list):
        prop_result = np.concatenate(prop_emb_list, axis=1)
        if self.svd:
            prop_result = get_embedding_dense(prop_result, self.dim)
        # prop_result = get_embedding_dense(prop_result, prop_result.shape[1])
        pos_glb = prop_result.mean(0)

        neg_prop_result = np.concatenate(neg_prop_emb_list, axis=1)
        if self.svd:
            neg_prop_result = get_embedding_dense(neg_prop_result, self.dim)
        # neg_prop_result = get_embedding_dense(neg_prop_result, neg_prop_result.shape[1])

        pos_info = sigmoid(pos_glb.dot(prop_result.T))
        neg_info = sigmoid(pos_glb.dot(neg_prop_result.T))
        pos_loss = np.mean(np.log(pos_info)).mean()
        neg_loss = np.mean(np.log(1 - neg_info)).mean()

        return -(pos_loss + neg_loss) / 2

    def __call__(self):
        search_space = self.build_search_space()
        trials = Trials()
        best = fmin(self.target_func, search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)

        # record search log
        # with open("search_.log", "w") as f:
        #     for trial in trials:
        #         f.write(str(trial) + "\n")

        best_result = self.prop(best)[0]
        best_result = np.concatenate(best_result, axis=1)
        print(f"best parameters: {best}")

        if self.svd:
            best_result = get_embedding_dense(best_result, self.dim)
        return best_result
