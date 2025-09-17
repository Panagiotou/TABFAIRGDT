import numpy as np
import pandas as pd


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from tabfairgdt.method import Method, proper, smooth
# global variables
from tabfairgdt import NUM_COLS_DTYPES, CAT_COLS_DTYPES
from collections import Counter

class FairCART(Method):
    def __init__(self, dtype, smoothing=False, proper=False, minibucket=5, random_state=None, max_depth=None, protected_arrays=None, verbose=False, criterion="dp", *args, **kwargs):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state
        self.fitted = False
        self.verbose = verbose
        
        self.criterion = f'fair_gini_{criterion}'

        if self.dtype in CAT_COLS_DTYPES:
            from custom_sklearn.tree import DecisionTreeClassifier as FairDecisionTreeClassifier
            self.cart = FairDecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=minibucket, random_state=self.random_state, criterion=self.criterion)#, class_weight={1: 1, 2: 5})

        if self.dtype in NUM_COLS_DTYPES:
            if self.verbose:
                print("\t\tRegression tree used")
            self.cart = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=minibucket, random_state=self.random_state)

        self.one_hot = False # We use trees from FARE, so no one-hot

    def fit(self, X_df, y_df, alpha=0, s=None, viz=False):
        if self.proper:
            X_df, y_df = proper(X_df=X_df, y_df=y_df, random_state=self.random_state)



        cat_columns = X_df.select_dtypes(include='category').columns
        # Get index positions of categorical columns
        # cat_pos = np.array([])
        # self.one_hot = True

        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=False, one_hot_cat_cols=self.one_hot)
        if self.dtype in NUM_COLS_DTYPES:
            self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)

        X = X_df
        y = y_df

        cat_pos = np.array([X.columns.get_loc(col) for col in cat_columns]).astype(np.int32)




        if self.dtype in CAT_COLS_DTYPES:
            # unique, counts = np.unique(y, return_counts=True)
            # print(dict(zip(unique, counts)))
            # exit(1)
            self.cart.fit(X, y, alpha=alpha, s=s, cat_pos=cat_pos)
        else:
            self.cart.fit(X, y)
            print("\tWarning, using normal Decission Tree Regressor (fair version not implemented yet)")


        # self.cart.fit(X, y)
        self.fitted = True

        # save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)

        leaves_y_df = pd.DataFrame({'leaves': leaves, 'y': y})

        self.leaves_y_dict = leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()
        
        leaves_y_index_df = pd.DataFrame({'leaves': leaves, 'index': list(range(len(X)))})

        self.leaves_y_index_dict = leaves_y_index_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()


        # print(self.leaves_y_dict)
        # sys.exit(1)

        # print(y_df.unique())
        # print(leaves[:5])
        # print(y[:5])
        # print(leaves_y_df.head())
        # print(leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()))
        # print(leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]))
        # print(leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).iloc[0])
        # print(leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).iloc[0].shape)
        # print(leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).iloc[1].shape)

        # print(self.leaves_y_dict)


        if viz:
            return X, y

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

            y_pred[indices] = np.random.choice(self.leaves_y_dict[leaf], size=len(indices), replace=True)

        if self.smoothing and self.dtype in NUM_COLS_DTYPES:
            y_pred = smooth(self.dtype, y_pred, self.y_real_min, self.y_real_max)

        return y_pred
