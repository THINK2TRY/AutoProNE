# AutoProNE: Automated and Unsupervised graph representation learning.
Code for **[Automated Unsupervised Graph Representation Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9547743)** in TKDE'21.

<h4>Examples:</h4>

```bash
python main.py --emb ./emb/wiki_deepwalk.embedding --adj ./emb/wikipedia.ungraph --saved-path ./out/wiki_spectral.embedding

python main.py --emb ./emb/cora_dgi.embedding --dataset cora --concat-search --prop-types sc ppr heat gaussian
```

* `--emb` : path of input embedding
* `--adj` : path of edgelist format adjacency matrix.
* `--concat-search` : search the filters used to concat, default is `True`.
* `--prop-types` : types of filters to use, options : `['ppr', 'heat', 'gaussian', 'sc']`.
* `--max-evals `: num of iterations of automl to optimize loss, default: 100.
* `--loss` : loss function used in AutoML searching, default: 'infomax'
* `--no-eval`: if set, do not evalute the embedding after propagation.
* `--workers` : the number of working threads in AutoML. default: 10. Try to set --workers to a larger number for faster training.  
* `--dataset` : (optional).
* `--saved-path` : path to save embeddings.


Currently using `optuna` as AutoML tool.



<h5> Dataset</h5>

The datasets used in the paper could be downloaded from this [link](https://cloud.tsinghua.edu.cn/f/546ee33df09c40cfa18a/?dl=1).



