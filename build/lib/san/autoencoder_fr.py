# pure implementation of SANs
# Skrlj, Dzeroski, Lavrac and Petkovic.

"""
The code containing neural network part, Skrlj 2019
"""

import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np

torch.manual_seed(123321)
np.random.seed(123321)

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class E2EDatasetLoader(Dataset):
    def __init__(self, features, targets=None):  # , transform=None
        self.features = features.tocsr()

        if targets is not None:
            self.targets = targets  # .tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())
        if self.targets is not None:
            target = torch.from_numpy(np.array(self.targets[index]))
        else:
            target = None
        return instance, target


def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown='ignore')
    return enc.fit_transform(lbx.reshape(-1, 1))


class SANNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size, dropout=0.02, num_heads=2, device="cuda"):
        super(SANNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.activation = nn.SELU()
        self.num_heads = num_heads
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_classes)
        self.multi_head = torch.nn.ModuleList([torch.nn.Linear(input_size, input_size) for _ in [1] * num_heads])

    def forward_attention(self, input_space, return_softmax=False):

        attention_output_space = []
        for head in self.multi_head:
            if return_softmax:
                attention_output_space.append(self.softmax(head(input_space)))
            else:
                # this is critical for maintaining a connection to the input space!
                attention_output_space.append(self.softmax(head(input_space)) * input_space)

        # initialize a placeholder
        placeholder = torch.zeros(input_space.shape).to(self.device)

        # traverse the heads and construct the attention matrix
        for element in attention_output_space:
            placeholder = torch.max(placeholder, element)

        # normalize by the number of heads
        out = placeholder  # / self.num_heads
        return out

    def get_mean_attention_weights(self):
        activated_weight_matrices = []
        for head in self.multi_head:
            wm = head.weight.data
            diagonal_els = torch.diag(wm)
            activated_diagonal = self.softmax2(diagonal_els)
            activated_weight_matrices.append(activated_diagonal)
        output_mean = torch.mean(torch.stack(activated_weight_matrices, axis=0), axis=0)

        return output_mean

    def forward(self, x):

        # attend and aggregate
        out = self.forward_attention(x)

        # dense hidden (l1 in the paper)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.activation(out)

        # dense hidden (l2 in the paper, output)
        out = self.fc3(out)
        return out

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True)

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()


class SAN:
    def __init__(self, batch_size=32, num_epochs=32, learning_rate=0.001, stopping_crit=10, hidden_layer_size=64,
                 dropout=0.2):  # , num_head=1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = torch.nn.CrossEntropyLoss()
        self.dropout = dropout
        self.batch_size = batch_size
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.num_params = None

    # not used:
    # def init_all(self, model, init_func, *params, **kwargs):
    #     for p in model.parameters():
    #         init_func(p, *params, **kwargs)

    def fit(self, features, labels):  # , onehot=False

        nun = len(np.unique(labels))
        logging.info("Found {} unique labels.".format(nun))
        train_dataset = E2EDatasetLoader(features, labels)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        stopping_iteration = 0
        current_loss = 10000
        self.model = SANNetwork(features.shape[1], num_classes=nun, hidden_layer_size=self.hidden_layer_size,
                                dropout=self.dropout, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Number of parameters {}".format(self.num_params))
        logging.info("Starting training for {} epochs".format(self.num_epochs))
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                logging.info("Stopping reached!")
                break
            losses_per_batch = []
            for i, (features, labels) in enumerate(dataloader):
                features = features.float().to(self.device)
                labels = labels.long().to(self.device)
                self.model.train()
                outputs = self.model.forward(features)
                outputs = outputs.view(labels.shape[0], -1)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))
            mean_loss = np.mean(losses_per_batch)
            if mean_loss < current_loss:
                current_loss = mean_loss
            else:
                stopping_iteration += 1
            logging.info("epoch {}, mean loss per batch {}".format(epoch, mean_loss))

    def predict(self, features, return_proba=False):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        with torch.no_grad():
            for features, _ in test_dataset:
                self.model.eval()
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        if not return_proba:
            a = [np.argmax(a_) for a_ in predictions]  # assumes 0 is 0
        else:
            a = []
            for pred in predictions:
                a.append(pred[1])

        return np.array(a).flatten()

    def predict_proba(self, features):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        a = [a_[1] for a_ in predictions]
        return np.array(a).flatten()

    def get_mean_attention_weights(self):
        return self.model.get_mean_attention_weights().detach().cpu().numpy()

    def get_instance_attention(self, instance_space):
        instance_space = torch.from_numpy(instance_space).float().to(self.device)
        return self.model.get_attention(instance_space).detach().cpu().numpy()

if __name__ == "__main__":

    from scipy import sparse
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier

    sns.set_style("whitegrid")
    dataobj = load_breast_cancer()
    X = dataobj['data']
    Y = dataobj['target']
    names = dataobj['feature_names']

    # let's overfit, just for demo purposes
    clf = SAN(num_epochs = 32, num_heads = 2, batch_size = 8, dropout = 0.2, hidden_layer_size = 32)
    X = sparse.csr_matrix(X)
    clf.fit(X, Y)
    preds = clf.predict(X)
    global_attention_weights = clf.get_mean_attention_weights()
    local_attention_matrix = clf.get_instance_attention(X.todense())
    mutual_information = mutual_info_classif(X,Y)
    rf = RandomForestClassifier().fit(X,Y).feature_importances_    

    plt.plot(names, global_attention_weights, label = "Global attention", marker = "x")
    plt.plot(names, np.mean(local_attention_matrix, axis = 0), label = "Local attention - mean", marker = "x")

    plt.plot(names, np.max(local_attention_matrix, axis = 0), label = "Local attention - max", marker = "x")

    plt.plot(names, mutual_information, label = "Mutual information", marker = ".")

    plt.plot(names, rf, label = "RandomForest", marker = ".")

    plt.legend(loc = 1)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.show()
