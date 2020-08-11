# ProNE++: Automated and Unsupervised graph representation learning.
mkdir out

Examples: 

python main.py --emb ./emb/wiki_deepwalk.embedding --adj ./emb/wikipedia.ungraph --saved-path ./out/wiki_spectral.embedding

python main.py --emb ./emb/cora_dgi.embedding --dataset cora --concat-search --prop-types sc ppr heat gaussian

* --emb : path of input embedding
* --adj : path of edgelist format adjacency matrix
* --concat-search : search the filters used to concat, action="store_true"
* --prop-types : types of filters to use, options : ['ppr', 'heat', 'gaussian', 'sc']
* --max-evals : iterations of automl to optimize loss, default=100
* --loss : loss function used in AutoML searching, default: 'infomax'
* --no-eval : If set, do not evalute the embedding after propagation.
* --dataset : (optional)
* --saved-path : path to save embeddings


Currently using `optuna` as AutoML tool.