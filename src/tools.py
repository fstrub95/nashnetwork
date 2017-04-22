__author__ = 'julien-perolat'
import numpy as np
import random as rd


def random_distr(l):
    r = rd.uniform(0, 1)
    s = 0
    for item, prob in l:
        s += prob
        if s >= r:
            return item
    return l[-1]


def sample_action_unif(action_set_list):
    action_unif_list = []
    for a_List in action_set_list:
        action_unif_list.append(rd.sample(a_List, 1)[0])
    return action_unif_list


def split(X, n):
    X_ = []
    while len(X) > n:
        X_.append(X[:n])
        X = X[n:]
    X_.append(X)
    return X_


def merge(X):
    X_ = []
    while len(X) > 0:
        X_ = X_ + X[0]
        X = X[1:]
    return X_


def merge_(X):
    X_ = []
    len_x = []
    for x in X:
        X_ = X_ + x
        len_x.append(len(x))
    return X_, len_x


def split_(X, len_x):
    X_ = []
    while len(len_x) > 0:
        X_.append(X[:len_x[0]])
        X = X[len_x[0]:]
        len_x = len_x[1:]
    return X_


import pickle

def save(obj, filename):
    print("Saving " + filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, 0)

def load_file(filename):
    with open(filename, 'rb') as f:
        v = pickle.load(f)
    return v
