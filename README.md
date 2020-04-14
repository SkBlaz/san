# Feature ranking with self-attention networks
This is the repository of the SAN paper, found here:

```
@misc{krlj2020feature,
    title={Feature Importance Estimation with Self-Attention Networks},
    author={Blaž Škrlj and Sašo Džeroski and Nada Lavrač and Matej Petkovič},
    year={2020},
    eprint={2002.04464},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Note that the full code with datasets to reproduce the paper can be found here: https://gitlab.com/skblaz/attentionrank (code is messy though, proceed with caution).

# Installing SANs
```
python setup.py install
```

or

```
pip install git+https://github.com/https://github.com/SkBlaz/san
```

# Using SANs
A simple usecase is given next:

```
    from sklearn.datasets import make_classification
    from scipy import sparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_breast_cancer
    import tqdm
    import san
    
    sns.set_style("whitegrid")
    dataobj = load_breast_cancer()
    X = dataobj['data']
    Y = dataobj['target']

    ## let's overfit, just for demo purposes
    clf = san.SAN(num_epochs = 18, num_head = 2, batch_size = 8, dropout = 0.05, hidden_layer_size = 32)
    X = sparse.csr_matrix(X)
    clf.fit(X, Y)
    preds = clf.predict(X)
    global_attention_weights = clf.get_mean_attention_weights()
    local_attention_matrix = clf.get_instance_attention(X.todense())

```