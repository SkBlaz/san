from scipy import sparse
from sklearn.datasets import load_breast_cancer
import san
import seaborn as sns


def test_simple_ranking():
    sns.set_style("whitegrid")
    dataobj = load_breast_cancer()
    x = dataobj['data']
    y = dataobj['target']
    # let's overfit, just for demo purposes
    clf = san.SAN(num_epochs=18, num_head=2, batch_size=8, dropout=0.05, hidden_layer_size=32)
    x = sparse.csr_matrix(x)
    clf.fit(x, y)
    predictions = clf.predict(x)
    global_attention_weights = clf.get_mean_attention_weights()
    local_attention_matrix = clf.get_instance_attention(x.todense())
    return predictions, global_attention_weights, local_attention_matrix


if __name__ == "__main__":
    test_simple_ranking()
