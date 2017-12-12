import argparse
import os.path as osp
import os
import numpy as np
from keras import metrics
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def import_files(path):
    data = list()
    if not osp.exists(path):
        print "Error. The file does not exist"
        return
    with open(path, "r") as f:
        for line in f:
            ll = line.strip().split(",")
            d = np.zeros(len(ll))
            for i in range(0, len(ll)):
                d[i] = float(ll[i])
            data.append(d)
    return data

def transform_data_format(data):
    num_features = len(data[0]) - 1
    num_sample = len(data)
    x = np.zeros((num_sample, num_features))
    y = np.zeros(num_sample)
    for i in xrange(0, num_sample):
        x[i] = data[i][0:-1]
        y[i] = data[i][-1]
    return x,y


def save_log(path, classifier, accuracy, precision, recall, f1):
    if not osp.exists(path):
        os.makedirs(path)
    filename = classifier+"_log.txt"
    header = ["Accuracy", "Precision", "Recall", "F1"]
    scores = [accuracy, precision, recall, f1]
    if not osp.exists(osp.join(path, filename)):
        with open(osp.join(path, filename), "w") as f:
            for (metric, value) in zip(header, scores):
                f.write(metric)
                f.write(": ")
                f.write(str(value))
                f.write("\n")


class RNN_clasifier:

    def __init__(self, x_train, y_train, x_test, y_test, units=16, dropout=0.2, recurrent_dropout=0.2):
        self._x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        self._y_train = y_train
        self._x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        self._y_test = y_test
        self._model = Sequential()
        self._model.add(LSTM(units=units, input_shape=self._x_train.shape[1:], dropout=dropout, recurrent_dropout=recurrent_dropout))
        self._model.add(Dense(1, activation='sigmoid'))
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, batch_size=10, num_epochs=10):
        self._model.fit(self._x_train, self._y_train, batch_size=batch_size, epochs=num_epochs)

    def test_model(self):
        score, accuracy = self._model.evaluate(self._x_test, self._y_test)
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        # preds = self._model.predict(self._x_test)
        # accuracy = accuracy_score(self._y_test, preds)
        # precision = precision_score(self._y_test, preds)
        # recall = recall_score(self._y_test, preds)
        # f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return accuracy, precision, recall, f1


def cross_validate_evaluation(x,y, n_folds, units=16, dropout=0.2,\
                              recurrent_dropout=0.2, batch_size=10, num_epochs=10):

    #skf = StratifiedKFold(y, n_splits=n_folds, shuffle=
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    skf.get_n_splits(x,y)
    accuracy = np.zeros(n_folds)
    precision = np.zeros(n_folds)
    recall = np.zeros(n_folds)
    f1 = np.zeros(n_folds)
    i = 0
    for train_index, test_index in skf.split(x, y):
        print "Running fold: %d / %d"%(i+1, n_folds)
        rnn = RNN_clasifier(x[train_index], y[train_index], x[test_index], y[test_index],\
                            units=units, dropout=dropout, recurrent_dropout=recurrent_dropout)
        rnn.train_model(batch_size=batch_size, num_epochs=num_epochs)
        accuracy[i], precision[i], recall[i], f1[i] = rnn.test_model()
        i = i + 1

    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/seb_class_begin_less.csv", type=str, help="path to the input data")
    parser.add_argument("--output_path", default="./results", type=str, help="path to the results")
    parser.add_argument("--num_fold", default=10, type=int,\
                        help="number of folds. determine the splitting strategy")
    parser.add_argument("--units", default=16, type=int, help="output dimension")
    parser.add_argument("--dropout", default=0.2, type=float,\
                        help="Float between 0 and 1. Fraction of the units to drop")
    parser.add_argument("--recurrent_dropout", default=0.2, type=float, \
                        help="Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.")
    parser.add_argument("--batch_size", default=10, type=int, help="Number of samples per gradient update")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs to train the model")

    args = parser.parse_args()

    data = import_files(args.input_path)
    x,y = transform_data_format(data)
    accuracy, precision, recall, f1 = cross_validate_evaluation(x, y, args.num_fold, args.units, args.dropout,\
                                                                args.recurrent_dropout, args. batch_size, \
                                                                args.num_epochs)

    save_log(args.output_path, "rnn", accuracy, precision, recall, f1)


if __name__ == '__main__':
    main()







