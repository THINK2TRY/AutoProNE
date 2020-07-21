import argparse
import time

from AutoSearch import AutoSearch, PlainFilter
from SearchFilter import SearchFilter
from spectral_prop import get_embedding_dense
from evaluate import evaluate
from utils import save_embedding


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--adj", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--saved-path", type=str, default="./out/prop_emb")
    parser.add_argument("--prop-types", nargs="+", default=["heat", "ppr", "gaussian", "sc"])
    parser.add_argument("--N", type=int, default=1000, help="Number of negative pairs sampled for auto-searching")
    parser.add_argument("--attention-search", action="store_true")
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--rescale", action="store_true", help="Rescale signals in Gaussian filter")
    parser.add_argument("--filter-search", action="store_true", help="Search an appropriate filter")
    t_args = parser.parse_args()
    if "dataset" not in t_args and "adj" not in t_args:
        raise ValueError("'adj' or 'dataset' is required")
    if t_args.dataset:
        t_args.adj = t_args.dataset + ".ungraph"
        t_args.label = t_args.dataset + ".cmty"
    else:
        rind = t_args.adj.rindex(".")
        t_args.label = t_args.adj[:rind] + ".cmty"
    return t_args


def main(args):
    out_path = args.saved_path

    if args.attention_search:
        model = AutoSearch(args)
    elif args.filter_search:
        model = SearchFilter(args)
    else:
        model = PlainFilter(args)

    print(" ---- start ----")
    start = time.time()

    spectral_emb = model()

    print(" ... # svd # ...")
    spectral_emb = get_embedding_dense(spectral_emb, spectral_emb.shape[1])

    end = time.time()
    print(" ---- end ----- ")
    print(f" .. time consumed: {end - start}s ...")

    # save the result
    print("... saving result ...")
    # save_npy_embedding(out_path, spectral_emb)
    save_embedding(out_path, spectral_emb)

    # evaluate ...
    print(" ... evaluating ...")
    evaluate(spectral_emb, label=args.label)


if __name__ == "__main__":
    # size = 1000000
    # dim = 128
    # emb = load_random_vector(size, dim)
    # mx = generate_adj_mx(size, size, size * 10)
    #
    # propagate(mx, emb)

    args = build_args()

    print(args)
    main(args)
