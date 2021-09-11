import argparse
import os
import time
import numpy as np

from plain_filter import PlainFilter
from concat_search import opConcatSearch
from evaluate import evaluate
from evaluate_nn import evaluate_pre
from utils import save_embedding


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--adj", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--saved-path", type=str, default="./out/")
    parser.add_argument("--prop-types", nargs="+", default=["heat", "ppr", "gaussian", "sc"])
    parser.add_argument("--N", type=int, default=1000, help="Number of negative pairs sampled for auto-searching")
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--concat-search", action="store_true")
    parser.add_argument("--loss", type=str, default="infomax")
    parser.add_argument("--no-svd", action="store_false")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--workers", type=int, default=10)

    t_args = parser.parse_args()
    if "dataset" not in t_args and "adj" not in t_args:
        raise ValueError("'adj' or 'dataset' is required")
    if t_args.dataset:
        t_args.adj = os.path.join("./data", t_args.dataset + ".ungraph")
        t_args.label = os.path.join("./data", t_args.dataset + ".cmty")
    else:
        rind = t_args.adj.rindex(".")
        t_args.label = t_args.adj[:rind] + ".cmty"
        t_args.dataset = t_args.adj[:rind].split("/")[-1]
    return t_args


def main(args):
    np.random.seed(0)
    out_path = args.saved_path
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        args.svd = False
    else:
        args.svd = True

    if args.concat_search:
        model = opConcatSearch(args)
    else:
        model = PlainFilter(args)

    print(args)

    print(" ---- start ----")
    start = time.time()
    spectral_emb = model()
    end = time.time()
    print(" ---- end ----- ")
    print(f" .. time consumed: {end - start}s ... ")

    # save the result
    print("... saving result ...")
    out_path = f"{out_path}_.emb"
    save_embedding(out_path, spectral_emb)

    # ======= evaluate ========
    if not args.no_eval:
        print(" ... evaluating ...")
        if args.dataset in ["cora", "citeseer", "pubmed"]:
            return evaluate_pre(args, spectral_emb)
        else:
            return evaluate(spectral_emb, label=args.label)


def enumerate_test():
    args = build_args()
    emb_map = {"ppi": "ppi", "blogcatalog": "blog", "wikipedia": "wiki", "dblp": "dblp", "youtube": "youtube"}
    f = open("nigh_res", "w")
    for dataset in ["ppi", "blogcatalog", "wikipedia", "dblp", "youtube"]:
        for alg in ["deepwalk", "line", "node2vec", "netmf", "hope"]:
            if dataset in ["dblp", "youtube"] and alg in ["netmf", "hope"]:
                continue
            ds = emb_map[dataset]
            emb_path = f"../net_emb/{alg}/{ds}_{alg}.embedding"
            adj_path = f"./data/{dataset}.ungraph"
            label_path = f"./data/{dataset}.cmty"
            for prop in ["sc", "ppr", "heat", "gaussian"]:
                print(f"{dataset}, {alg}, {prop}")
                args.prop_types = [prop]
                args.emb = emb_path
                args.adj = adj_path
                args.label = label_path
                res = main(args)
                f.write(f"-- {dataset} -- {alg} -- {prop} -- \n")
                for key, val in res.items():
                    f.write(f"{key} : {val} \n")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    args = build_args()
    main(args)
