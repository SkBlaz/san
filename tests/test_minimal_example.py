from scipy import sparse
from sklearn.datasets import load_breast_cancer
import san

def test_simple_ranking():
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


if __name__ == "__main__":

    test_simple_ranking()
