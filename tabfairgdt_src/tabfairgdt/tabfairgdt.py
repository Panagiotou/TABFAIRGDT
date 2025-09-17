import numpy as np
import pandas as pd

# classes
from tabfairgdt.validator import Validator
from tabfairgdt.processor import Processor
# global variables
from tabfairgdt import NUM_COLS_DTYPES
from tabfairgdt.processor import NAN_KEY
from tabfairgdt.method import CART_LEAF_RELAB_LAMDA, METHODS_MAP, NA_METHODS, NORMAL_CART
import sys
from tqdm import tqdm

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np


class TABFAIRGDT:
    def __init__(self,
                 method=None,
                 visit_sequence=None,
                 # predictor_matrix=None,
                 proper=False,
                 cont_na=None,
                 smoothing=False,
                 default_method=NORMAL_CART,
                 numtocat=None,
                 catgroups=None,
                 seed=None,
                 dtype_map=None,
                 verbose=False,
                 protected_attribute=None,
                 target=None,
                 target_only_fair=False,
                 criterion="dp",
                 re_order=False,
                 acc_threshold=-1,
                 parallel=True
                ):
        # initialise the validator and processor
        self.validator = Validator(self)
        self.processor = Processor(self)

        # initialise arguments
        self.method = method
        self.visit_sequence = visit_sequence
        self.predictor_matrix = None
        self.proper = proper
        self.cont_na = cont_na
        self.smoothing = smoothing
        self.default_method = default_method
        self.numtocat = numtocat
        self.catgroups = catgroups
        self.seed = seed
        self.dtype_map = dtype_map
        self.verbose = verbose
        self.protected_attribute = protected_attribute
        self.target = target
        self.target_only_fair = target_only_fair
        self.criterion = criterion
        self.acc_threshold = acc_threshold
        self.parallel = parallel
        # check init
        self.validator.check_init()
        self.re_order = re_order

        self.df_dtypes = dtype_map
 

        self.standard_method_args = {
            "proper":self.proper, 
            "random_state": self.seed,
            "verbose":verbose,
        }


    def is_fitted(self, estimator):
        return estimator.fitted

    def encode(self, df):
        df_encoded = df.copy()
        for column in df_encoded.columns:
            dtype = self.dtype_map[column]
            if dtype == "category":
                df_encoded[column] = df_encoded[column].map(self.reverse_mappings[column]).astype("category")
        return df_encoded

    def decode(self, df):
        df_decoded = df.copy()
        for column, mapping in self.original_mappings.items():
            if column in df_decoded.columns:
                df_decoded[column] = df_decoded[column].map(mapping)
        return df_decoded
        

    def update_mappings(self, df):
        self.original_mappings = {}
        for column, dtype in self.dtype_map.items():
            if dtype == "category":
                self.original_mappings[column] = dict(zip(range(1, len(df[column].unique())+1), df[column].unique()))

        self.reverse_mappings = {col: {v: k for k, v in self.original_mappings[col].items()} for col in self.original_mappings}

    def get_corr(self, df, target_col="target"):
        """
        Get features sorted by correlation with target column.
        Handles both continuous and discrete features.
        
        Parameters:
        df: DataFrame containing the features
        target_col: str, either "target" or "protected" to specify which column to correlate with
        
        Returns:
        pandas.Series: Features sorted by absolute correlation value (descending)
        """
        # Determine which target column to use
        if target_col == "target":
            target_series = self.target_col
        elif target_col == "protected":
            target_series = self.protected_col
        else:
            raise ValueError("target_col must be either 'target' or 'protected'")
        
        correlations = {}
        
        for column in df.columns:
            # For numeric columns, use Pearson correlation
            if df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                corr = df[column].corr(target_series)
            else:
                # For discrete/categorical columns, convert to numeric codes
                # This works for both string categories and already-encoded discrete variables
                if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                    numeric_col = pd.Categorical(df[column]).codes
                else:
                    numeric_col = df[column]
                
                corr = pd.Series(numeric_col).corr(target_series)
            
            if not pd.isna(corr):
                correlations[column] = corr
            else:
                correlations[column] = 0
                    

        
        # Convert to Series and sort by absolute correlation value (descending)
        corr_series = pd.Series(correlations)
        sorted_correlations = corr_series.sort_values(ascending=False)
        
        # Return the correlation values (with original signs) in sorted order
        return corr_series[sorted_correlations.index]
    
    def fit(self, df, lamda=0.5):
        # TODO check df and check/EXTRACT dtypes
        # - all column names of df are unique
        # - all columns data of df are consistent
        # - all dtypes of df are correct ('int', 'float', 'datetime', 'category', 'bool'; no object)
        # - can map dtypes (if given) correctly to df
        # should create map col: dtype (self.df_dtypes)

        # train_df_encoded = dataset_generator.encode(train_df)
        # new_order = ['age', 'workclass', 'education', 'marital-status', 'occupation',
        #        , 'race', 'sex', 'relationship', 'capital-gain', 'capital-loss',
        #        'hours-per-week', 'native-country', 'Class-label']

        # train_df = train_df.reindex(columns=new_order)

        self.lamda = lamda

        self.original_df_cols = list(df.columns)


        self.update_mappings(df)


        df = self.encode(df)

        self.target_col = df[self.target].copy()
        self.protected_col = df[self.protected_attribute].copy()
        self.protected_arrays = df[[self.protected_attribute]].to_numpy().astype(float)


        df.drop(self.target, axis=1, inplace=True)
        df.drop(self.protected_attribute, axis=1, inplace=True)

        if self.re_order:
            print("order changed from:")
            print("\t", list(df.columns))
            print("to:")

            if self.re_order == "corr_asc_target":

                correlations = self.get_corr(df, target_col="target")

                sorted_features = correlations.sort_values(ascending=False).index.tolist()

            elif self.re_order == "corr_desc_target":

                correlations = self.get_corr(df, target_col="target")

                sorted_features = correlations.sort_values(ascending=True).index.tolist()

            elif self.re_order == "corr_asc_protected":

                correlations = self.get_corr(df, target_col="protected")

                sorted_features = correlations.sort_values(ascending=False).index.tolist()

            elif self.re_order == "corr_desc_protected":

                correlations = self.get_corr(df, target_col="protected")

                sorted_features = correlations.sort_values(ascending=True).index.tolist()
            else:
                raise RuntimeError("Reordering not defined")
            
            df = df[sorted_features]
            print("\t", list(df.columns))
            
        else:
            print("No re-ordering of features, using visit sequence as is")



        

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)

        # check processor
        self.validator.check_processor()
        # preprocess
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)


        self.cat_pos = list(self.original_mappings.keys())
        self.cat_pos.remove(self.target)

        # check fit
        self.validator.check_fit()
        # fit



        original_numeric =  processed_df.select_dtypes(['float', 'integer'])
        original_discrete = processed_df.select_dtypes(['object', 'category'])
        if self.verbose:
            print("Numeric cols", list(original_numeric.columns))
            print("Discrete cols", list(original_discrete.columns))


        self.method_args_all_cols = {}
        
        for col in self.df_columns:
            self.method_args_all_cols[col] = self.standard_method_args.copy()

        if self.parallel:
            self._fit_parallel(processed_df)
        else:
            self._fit(processed_df)



    def _fit_parallel(self, df):
        self.saved_methods = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        cat_indices = np.array([idx for idx, col in enumerate(df.columns) if col in self.cat_pos]).astype(np.int32)

        # Define function to fit a single column
        def fit_column(col, visit_step):
            if self.verbose:
                print('train_{}'.format(col))

            if col == self.target:
                method_args_target = self.method_args_all_cols.copy()
                method_args_target["fair"] = True
                method_args_target["acc_threshold"] = self.acc_threshold

                target_method = METHODS_MAP['cart_leaf_relab_lamda'](
                    dtype='category', smoothing=False, **method_args_target
                )
                target_method.fit(X_df=df, y_df=self.target_col, s=self.protected_arrays, cat_pos=cat_indices, alpha=self.lamda)

                return col, target_method

            elif col == self.protected_attribute:
                method_args_protected = self.method_args_all_cols.copy()

                protected_method = METHODS_MAP['cart'](dtype='category', smoothing=False, **method_args_protected)
                protected_method.fit(X_df=df, y_df=self.protected_col)

                return col, protected_method

            else:
                col_method = METHODS_MAP[self.method[col]](
                    dtype=self.df_dtypes[col], smoothing=self.smoothing[col], **self.method_args_all_cols[col]
                )
                col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]

                if self.method[col] != "sample" and self.is_fitted(col_method):
                    print("already fitted")
                    exit(1)

                col_method.fit(X_df=df[col_predictors], y_df=df[col])

                return col, col_method

        self.visit_sequence_all = self.visit_sequence.copy()
        self.visit_sequence_all[self.protected_attribute] = len(self.visit_sequence)
        self.visit_sequence_all[self.target] = len(self.visit_sequence)

        # Run in parallel
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(fit_column)(col, visit_step) 
            for col, visit_step in tqdm(self.visit_sequence_all.sort_values().items(), total=len(self.visit_sequence_all), desc="Fitting CART")
        )

        # Store results in saved_methods
        self.saved_methods = {col: method for col, method in results}



    def _fit_old(self, df):
        self.saved_methods = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()


        for col, visit_step in tqdm(self.visit_sequence.sort_values().items(), total=len(self.visit_sequence), desc="Fitting CART"):
            if self.verbose:
                print('train_{}'.format(col))

            # initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], smoothing=self.smoothing[col], **self.method_args_all_cols[col])
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            
            
            if self.method[col] != "sample" and self.is_fitted(col_method):
                print("already fitted")
                exit(1)

            col_method.fit(X_df=df[col_predictors], y_df=df[col])

            # save the method
            self.saved_methods[col] = col_method



        cat_indices = np.array([idx for idx, col in enumerate(df.columns) if col in self.cat_pos]).astype(np.int32)

        method_args_target = self.method_args_all_cols.copy()
        method_args_target["fair"] = True
        method_args_target["acc_threshold"] = self.acc_threshold


        target_method = METHODS_MAP['cart_leaf_relab_lamda'](dtype='category', smoothing=False, **method_args_target)
        target_method.fit(X_df=df, y_df=self.target_col, s=self.protected_arrays, cat_pos=cat_indices, alpha=self.lamda)

        self.saved_methods[self.target] = target_method


        method_args_protected = self.method_args_all_cols.copy()

        protected_method = METHODS_MAP['cart'](dtype='category', smoothing=False, **method_args_protected)
        protected_method.fit(X_df=df, y_df=self.protected_col)

        self.saved_methods[self.protected_attribute] = protected_method


    def generate(self, k=None):
        self.k = k

        # check generate
        self.validator.check_generate()
        # generate
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df)

        processed_synth_df = self.decode(processed_synth_df)

        processed_synth_df = processed_synth_df[self.original_df_cols]

        return processed_synth_df

    def _generate(self):
        synth_df = pd.DataFrame(data=np.zeros([self.k, len(self.visit_sequence)]), columns=self.visit_sequence.index)

        for col, visit_step in tqdm(self.visit_sequence.sort_values().items(), total=len(self.visit_sequence), desc="Generating from CART"):
            if self.verbose:
                print('generate_{}'.format(col))

            # reload the method
            col_method = self.saved_methods[col]
            # predict with the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]

            synth_df[col] = col_method.predict(synth_df[col_predictors])
            # profiler.print()
            # change all missing values to 0
            if col in self.processor.processing_dict[NAN_KEY] and self.df_dtypes[col] in NUM_COLS_DTYPES and self.method[col] in NA_METHODS:
                nan_indices = synth_df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                synth_df.loc[nan_indices, col] = 0

            # map dtype to original dtype (only excpetion if column is full of NaNs)
            if synth_df[col].notna().any():
                synth_df[col] = synth_df[col].astype(self.df_dtypes[col])


        target_method = self.saved_methods[self.target]
        target_column = target_method.predict(synth_df[list(self.visit_sequence.index)])


        protected_method = self.saved_methods[self.protected_attribute]
        protected_column = protected_method.predict(synth_df[list(self.visit_sequence.index)])


        synth_df[self.target] = target_column
        synth_df[self.protected_attribute] = protected_column

        return synth_df
