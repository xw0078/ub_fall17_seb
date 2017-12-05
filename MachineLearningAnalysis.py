import argparse
import os.path as osp
import os
import h5py
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score



def import_files(path):
    f = h5py.File(path, 'r')
    file = f['Data']
    group_keys = list(file.keys())
    data = list()
    for key in group_keys:
        dataset_i = list(file[group_keys[0]])
        for d in dataset_i:
            data.append(d)
    return data


def transform_data_format(data):
    num_features = len(data[0]) - 1
    num_sample = len(data)
    x = np.zeros((num_sample, num_features))
    y = np.zeros(num_sample)
    for i in xrange(0, num_sample):
        x[i] = data[i,0:-1]
        y[i] = data[i,-1]
    return x,y


def save_log(path, classifier, accuracy, precision, recall, f1):
    if not osp.exists(path):
        os.makedirs(path)
    filename = classifier+"_log.csv"
    header = ["Accuracy", "Precision", "Recall", "F1"]
    scores = [accuracy, precision, recall, f1]
    if not osp.exists(osp.join(path, filename)):
        with open(osp.join(path, filename), "w") as f:
            f.write(",".join(header) + "\n")
            f.write(",".join(scores)+"\n")


class Classifiers:
    def __init__(self, x, y, num_fold):
        self._x = x
        self._y = y
        self._num_fold = num_fold
        self._SVM = None
        self._LR = None
        self._NB = None
        self._MLP = None
        self._KNN = None

    def svm_predict(self, C=1.0, kernel='rbf', gamma=0.3):
        self._SVM = SVC(C=C, kernel=kernel, gamma=gamma)
        preds = cross_val_predict(self._SVM, self._x, self._y)
        return preds

    def lr_predict(self, penalty='l2', C=1.0):
        self._LR = LogisticRegression(penalty=penalty, C=C)
        preds = cross_val_predict(self._LR, self._x, self._y)
        return preds

    def nb_predict(self, priors=None):
        self._NB = GaussianNB(priors=priors)
        preds = cross_val_predict(self._NB, self._x, self._y)
        return preds

    def mlp_predict(self, hidden_layer_sizes=100, activation='relu',\
                 solver='adam', alpha=0.0001, learning_rate='constant',\
                 learning_rate_init= 0.001, power_t=0.5, momentum=0.9,\
                 beta_1=0.9, beta_2 =0.999, epsilon=1e-8):
        self._MLP = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,\
                                  solver=solver, alpha=alpha, learning_rate=learning_rate,\
                                  learning_rate_init=learning_rate_init,power_t=power_t,\
                                  momentum=momentum,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
        preds = cross_val_predict(self._MLP, self._x, self._y)
        return preds

    def knn_predict(self, n_neighbors=5):
        self._KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        preds = cross_val_predict(self._KNN, self._x, self._y)
        return preds

    def evaluation(self, preds):
        accuracy = accuracy_score(self._y, preds)
        precision = precision_score(self._y, preds)
        recall = recall_score(self._y, preds)
        f1 = 2 * (precision * recall) / (precision + recall)
        return accuracy, precision, recall, f1

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/data.h5", type=str, help="path to the input data")
    parser.add_argument("--out_path", default="./results", type=str, help="path to the results")
    parser.add_argument("--classifier", default="lr", type=str,\
                        help="classifier type. it must be one of 'svm', 'lr', 'nb', 'mlp', 'knn'")
    parser.add_argument("--num_fold", default=10, type=int,\
                        help="number of folds. determine the splitting strategy")
    parser.add_argument("--C", default=1.0, type=float, help="penalty parameter")
    parser.add_argument("--kernel", default="rbf", type=str,\
                        help="kernel type. it must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’.\
                         only used when classifier is svm ")
    parser.add_argument("--gamma", default=0.3, type=float, \
                        help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. only used when classifer is svm")
    parser.add_argument("--penalty", default="l1", type=str, help=" ‘l1’ or ‘l2’. only used when classifier is lr")
    parser.add_argument("--hidden_layer_sizes", default=100, type=int, help="only used when classifier is mlp")
    parser.add_argument("--activation", default="relu", type=str, \
                        help="‘identity’, ‘logistic’, ‘tanh’, ‘relu’. only used when classifier is mlp")
    parser.add_argument("--solver", default="adam", type=str,\
                        help="‘lbfgs’, ‘sgd’, ‘adam’. only used when classifier is mlp and learning rate is adaptive")
    parser.add_argument("--learning_rate", default='constant', type=str,\
                        help="‘constant’, ‘invscaling’, ‘adaptive’. only used when classifier is mlp")
    parser.add_argument("--learning_rate_init", default=0.001, type=float,\
                        help="the initial learning rate used. only used when classifier is mlp")
    parser.add_argument("--alpha", default=0.0001, type=float,\
                        help="L2 penalty parameter, only used when classifier is mlp")
    parser.add_argument("--momentum", default=0.9, type=float,\
                        help="momentum for gradient descent update. should be between 0 and 1.\
                         only used when classifier is mlp and solver is sgd")
    parser.add_argument("--power_t", default=0.5, type=float, \
                        help="the exponent for inverse scaling learning rate. only used when \
                        classifier is mlp, learning rate is invscaling and solver is sgd.")
    parser.add_argument("beta_1", default=0.9, type=float,\
                        help="between 0 and 1. only used when solver is adam")
    parser.add_argument("beta_2", default=0.999, type=float,\
                        help="between 0 and 1. only used when solver is adam")
    parser.add_argument("--epsilon", default= 1e-8, type=float,\
                        help="only used when solver is adam")
    args = parser.parse_args()

    data = import_files(args.input_path)
    x,y = transform_data_format(data)
    cc = Classifiers()







