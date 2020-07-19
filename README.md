# ProNE_PlusPlus
mkdir out
python main.py --emb ./emb/wiki_deepwalk.embedding --adj ./emb/wikipedia.ungraph --saved-path ./out/wiki_spectral.embedding

--emb : path of input embedding
--adj : path of edgelist format adjacency matrix
--saved-path : path to save embeddings
--search : whether to use auto-ml or not
--prop-types : types of filters to use, options : ['ppr', 'heat', 'gaussian', 'sc']