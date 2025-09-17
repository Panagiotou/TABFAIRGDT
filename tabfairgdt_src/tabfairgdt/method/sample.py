import numpy as np

from tabfairgdt.method import Method, proper, smooth
# global variables
from tabfairgdt import NUM_COLS_DTYPES



class SampleMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, random_state=None, *args, **kwargs):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state

    def fit(self, y_df=None, *args, **kwargs):
        if self.proper:
            y_df = proper(y_df=y_df)
        if self.dtype in NUM_COLS_DTYPES:
            self.x_real_min, self.x_real_max = np.min(y_df), np.max(y_df)

        self.values = y_df.to_numpy()

        self.unique = np.unique(self.values)

    def predict(self, X_test_df, re_order=False):
        
        # np.random.seed(self.random_state)
        n = X_test_df.shape[0]


        if re_order:
            y_pred = np.sort(np.random.choice(self.unique, size=n, replace=True))
            return y_pred

        y_pred = np.random.choice(self.values, size=n, replace=True)

        if self.smoothing and self.dtype in NUM_COLS_DTYPES:
            y_pred = smooth(self.dtype, y_pred, self.x_real_min, self.x_real_max)

        return y_pred
