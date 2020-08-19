import optuna
import numpy as np
import math
import random
import scipy.sparse as sp

from spectral_propagation import get_embedding_dense, propagate
from utils import load_embedding, load_adjacency_mx, sigmoid


class opConcatSearch(object):
    def __init__(self, args):
        self.prop_types = args.prop_types
        self.max_evals = args.max_evals
        self.svd = args.svd
        self.loss_type = args.loss
        self.n_workers = args.workers

        # load adjacency matrix and raw embedding
        self.emb = load_embedding(args.emb)
        self.adj, self.num_nodes, self.num_edges = load_adjacency_mx(args.adj)
        self.laplacian = None
        self.dim = self.emb.shape[1]

        assert self.num_nodes == self.emb.shape[0]
        # negative pairs
        self.negative_pairs = int(math.sqrt(self.num_edges))

        if self.loss_type == "infonce":
            self.batch_size = 64
            neg_index = []
            for i in range(self.num_nodes):
                select = np.random.choice(self.num_nodes, self.batch_size, replace=False)
                while i in select:
                    select = np.random.choice(self.num_nodes, self.batch_size, replace=False)
                neg_index.append(select)
            self.neg_index = np.array(neg_index)
            self.neg_emb = self.emb[self.neg_index]
        elif self.loss_type == "infomax":
            # self.permutation = np.random.permutation(np.arange(self.num_nodes))
            pass

    def build_search_space(self, trial):
        space = {}
        for f in self.prop_types:
            space[f] = trial.suggest_categorical(f, [0, 1])
        if space.get("heat", 0) == 1:
            space["t"] = trial.suggest_uniform("t", 0.1, 0.9)
        if space.get("gaussian", 0) == 1:
            space["mu"] = trial.suggest_uniform("mu", 0.1, 2)
            space["theta"] = trial.suggest_uniform("theta", 0.2, 1.5)
        if space.get("ppr", 0) == 1:
            space["alpha"] = trial.suggest_uniform("alpha", 0.2, 0.8)
        return space

    def prop(self, params):
        prop_types = [key for key, value in params.items() if value == 1 and key in self.prop_types]
        if not prop_types:
            print(" -- dropped -- ")
            return None, None

        prop_result_list = []
        for selected_prop in prop_types:
            prop_result = propagate(self.adj, self.emb, selected_prop, params)
            prop_result_list.append(prop_result)

        if self.loss_type == "infomax":
            neg_prop_result = []
            pmt = self.permutation
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

    def target_func(self, trial):
        params = self.build_search_space(trial)
        self.permutation = np.random.permutation(np.arange(self.num_nodes))
        prop_result_emb, neg_prop_result_emb = self.prop(params)
        if prop_result_emb is None:
            return 100
        if self.loss_type == "infomax":
            loss = self.infomax_loss(prop_result_emb, neg_prop_result_emb)
        elif self.loss_type == "infonce":
            loss = self.infonce_loss(prop_result_emb)
        else:
            loss = self.sparse_cut_loss(prop_result_emb)
        return loss

    def sparse_cut_loss(self, prop_emb_list):
        if not self.laplacian:
            degree = self.adj.sum(1).todense()
            degree_matrix = sp.csc_matrix(
                (degree, (np.arange(self.num_nodes), np.arange(self.num_nodes))),
                shape=(self.num_nodes, self.num_nodes))
            self.laplacian = degree_matrix - self.adj

        # Sparest Cut Loss
        prop_emb = np.concatenate(prop_emb_list, axis=1)
        if self.svd:
            prop_emb = get_embedding_dense(prop_emb, self.dim)
        pairs = np.random.choice(self.num_nodes, (2, self.negative_pairs))
        neg_emb = prop_emb[pairs]
        loss = np.sum((neg_emb[0] - neg_emb[1]) ** 2, axis=1).mean()
        return 1. / loss

    def infonce_loss(self, prop_emb_list, *args, **kwargs):
        T = 0.07

        pos_infos = []
        for smoothed in prop_emb_list:
            pos_info = np.exp(np.sum(smoothed * self.emb, -1) / T)
            assert pos_info.shape == (self.num_nodes,)
            pos_infos.append(pos_info)

        neg_infos = []
        for idx, smoothed in enumerate(prop_emb_list):
            neg_info = np.exp(np.sum(np.tile(smoothed[:, np.newaxis, :], (1, self.batch_size, 1)) * self.neg_emb, -1) / T).sum(-1)
            assert neg_info.shape == (self.num_nodes,)
            neg_infos.append(neg_info + pos_infos[idx])

        pos_neg = np.array(pos_infos) / np.array(neg_infos)
        if np.isnan(pos_neg).any():
            pos_neg = np.nan_to_num(pos_neg)

        loss = -np.log(pos_neg).mean()
        return loss/10

    def infomax_loss(self, prop_emb_list, neg_prop_emb_list):
        prop_result = np.concatenate(prop_emb_list, axis=1)
        if self.svd:
            prop_result = get_embedding_dense(prop_result, self.dim)
        pos_glb = prop_result.mean(0)
        pos_info = sigmoid(pos_glb.dot(prop_result.T))
        pos_loss = np.mean(np.log(pos_info)).mean()

        neg_loss = 0
        neg_step = 1
        for _ in range(neg_step):
            neg_prop_result = np.concatenate(neg_prop_emb_list, axis=1)
            if self.svd:
                neg_prop_result = get_embedding_dense(neg_prop_result, self.dim)

            neg_info = sigmoid(pos_glb.dot(neg_prop_result.T))
            neg_loss += np.mean(np.log(1 - neg_info)).mean()
            random.shuffle(neg_prop_emb_list)

        return -(pos_loss + neg_loss) / (1 + neg_step)

    def __call__(self):
        study = optuna.create_study()
        study.optimize(self.target_func, n_jobs=self.n_workers, n_trials=self.max_evals)

        best_params = study.best_params

        best_result = self.prop(best_params)[0]
        best_result = np.concatenate(best_result, axis=1)
        print(f"best parameters: {best_params}")

        if self.svd:
            best_result = get_embedding_dense(best_result, self.dim)
        return best_result
