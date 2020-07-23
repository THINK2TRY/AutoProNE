# ProNE_PlusPlus
mkdir out

python main.py --emb ./emb/wiki_deepwalk.embedding --adj ./emb/wikipedia.ungraph --saved-path ./out/wiki_spectral.embedding

* --emb : path of input embedding
* --adj : path of edgelist format adjacency matrix
* --saved-path : path to save embeddings
* --concat-search : search the filters used to concat, action="store_true"
* --contra-search : search one filter, action="store_true"
* --prop-types : types of filters to use, options : ['ppr', 'heat', 'gaussian', 'sc']