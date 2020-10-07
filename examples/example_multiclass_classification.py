## mcc

from scipy import sparse
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris
import san
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif  # chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import tqdm
from sklearn.model_selection import KFold

data_obj = load_iris()
x = data_obj['data']
y = data_obj['target']
names = data_obj['feature_names']

clf = san.SAN(num_epochs=30, num_heads=4, batch_size=16, dropout=0.2, hidden_layer_size=32, stopping_crit = 10, learning_rate = 0.01)

kf = KFold(n_splits=10)
accuracy_results = []
for train_index, test_index in kf.split(x):
    train_x = x[train_index]
    test_x = x[test_index]
    train_y = y[train_index]
    test_y = y[test_index]
    x_sp = sparse.csr_matrix(train_x)
    xt_sp = sparse.csr_matrix(test_x)
    clf.fit(x_sp, train_y)
    predictions = clf.predict(xt_sp)
    score = accuracy_score(predictions, test_y)
    accuracy_results.append(score)
    
    # uncomment to get importances
    # global_attention_weights = clf.get_mean_attention_weights()
    # local_attention_matrix = clf.get_instance_attention(x)
    
print("Accuracy (iris dataset) {} ({})".format(np.mean(accuracy_results), np.std(accuracy_results)))
