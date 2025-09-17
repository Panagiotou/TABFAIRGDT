import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from tabfairgdt.method import Method, proper, smooth
# global variables
from tabfairgdt import NUM_COLS_DTYPES, CAT_COLS_DTYPES
from collections import Counter
from sklearn.metrics import accuracy_score
from tabfairgdt.method import _lamda_relabeling

class FairCARTLeafRelabLamda(Method):
    def __init__(self, dtype, smoothing=False, proper=False, minibucket=5, random_state=None, max_depth=None, protected_arrays=None, verbose=False, fair=True, criterion="dp", acc_threshold=-1, *args, **kwargs):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state
        self.fitted = False
        self.verbose = verbose
        self.fair = fair
        self.acc_threshold = acc_threshold

        self.cart = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=minibucket, random_state=self.random_state)
        self.one_hot = True

    def get_leaves_to_relabel(self, X, y, y_pred, s=None, round_value=10, disc_threshold=0, acc_threshold=-1):
        leaves_relabel = _lamda_relabeling.leaves_to_relabel(self.cart, X, y, y_pred, s, acc_threshold=acc_threshold, disc_threshold=disc_threshold)
        return list(leaves_relabel)


    # def relabel_tree_test(self, X, y, y_pred, s=None, round_value=10, threshold=0):

    #     accuracy = round(accuracy_score(y, y_pred), round_value)
    #     discrimination = round(_relabeling.discrimination_dataset(y_pred, s), round_value)
    #     leaves_relabel = _relabeling.leaves_to_relabel(self.cart, X, y, y_pred, s, threshold)

    #     sum_acc = 0
    #     sum_disc = 0
    #     for leaf in leaves_relabel:
    #         sum_acc += leaf.acc
    #         sum_disc += leaf.disc

    #     sum_acc = round(sum_acc, round_value)
    #     sum_disc = round(sum_disc, round_value)
    #     _relabeling.relabeling(self.cart, X, y, y_pred, s, threshold)
    #     y_pred_relabel = self.cart.predict(X)
    #     accuracy_relabel = round(accuracy_score(y, y_pred_relabel), round_value)
    #     discrimination_relabel = round(_relabeling.discrimination_dataset(y_pred_relabel, s),
    #                                 round_value)
    #     new_acc = round(accuracy + sum_acc, round_value)
    #     new_disc = round(discrimination + sum_disc, round_value)

    #     print("Old accuracy", accuracy, "New accuracy", new_acc)
    #     print("old discrimination", discrimination, "new discrimination", new_disc)
    #     exit(1)

    def fit(self, X_df, y_df, alpha=0, s=None, cat_pos=None, viz=False):
        if self.proper:
            X_df, y_df = proper(X_df=X_df, y_df=y_df, random_state=self.random_state)


        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=False, one_hot_cat_cols=self.one_hot)
        if self.dtype in NUM_COLS_DTYPES:
            self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)

        X = X_df
        y = y_df
        self.cart.fit(X, y)


        y_pred = self.cart.predict(X)
        y_ = np.array(y.map({1: 0, 2: 1}).tolist())
        y_pred_ = np.array(list(map(lambda x: 0 if x == 1 else 1, y_pred)))
        s_ = np.array(list(map(lambda x: 0 if x[0] == 1 else 1, s)))

        leaves_to_relabel = self.get_leaves_to_relabel(X, y_, y_pred_, s=s_, acc_threshold=self.acc_threshold)


        leave_ids_to_relabel = [l.node_id for l in leaves_to_relabel]



        # self.cart.fit(X, y)
        self.fitted = True

        # save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)

        # print(leaves)
        # print(leaves_to_relabel)
        # exit(1)

        leaves_y_df = pd.DataFrame({'leaves': leaves, 'y': y})

        leaves_y_dict = leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()

        self.leaves_y_probs_dict = {}

        def adjust_probs(probs, lamda=0):
            max_cls = max(probs, key=probs.get)  # Find class with max probability
            p = probs[max_cls]
            new_p = p * (1 - lamda) + (1 - p) * lamda  # Apply soft switch
            probs[max_cls] = new_p
            other_cls = 1 if max_cls == 2 else 2
            probs[other_cls] = 1 - new_p  # Ensure sum to 1
            return probs

        clss = y.unique()
        # Loop through each key-value pair in the original dictionary
        for leaf, value in leaves_y_dict.items():
            total = len(value)  # Total number of elements in the array

            weighted_probs = {item: np.sum(value == item) / total for item in np.unique(value)}
            weighted_probs = {cls: weighted_probs.get(cls, 0.0) for cls in clss}


            if leaf in leave_ids_to_relabel:
                weighted_probs = adjust_probs(weighted_probs, lamda=alpha)


            self.leaves_y_probs_dict[leaf] = weighted_probs

    def predict(self, X_test_df):
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=False, one_hot_cat_cols=self.one_hot, fit=False)

        # predict the leaves and for each leaf randomly sample from the observed values
        # X_test = X_test_df.to_numpy()
        X_test = X_test_df
        leaves_pred = self.cart.apply(X_test)
        y_pred = np.zeros(len(leaves_pred), dtype=object)

        leaves_pred_index_df = pd.DataFrame({'leaves_pred': leaves_pred, 'index': range(len(leaves_pred))})
        leaves_pred_index_dict = leaves_pred_index_df.groupby('leaves_pred').apply(lambda x: x.to_numpy()[:, -1]).to_dict()

        for leaf, indices in leaves_pred_index_dict.items():

            keys = np.array(list(self.leaves_y_probs_dict[leaf].keys()))
            probabilities = np.array(list(self.leaves_y_probs_dict[leaf].values()))

            y_pred[indices] = np.random.choice(keys, size=len(indices), p=probabilities)

        if self.smoothing and self.dtype in NUM_COLS_DTYPES:
            y_pred = smooth(self.dtype, y_pred, self.y_real_min, self.y_real_max)

        return y_pred
