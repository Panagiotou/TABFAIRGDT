import numpy as np
import pandas as pd

# classes
from tabfairgdt.validator import Validator
from tabfairgdt.processor import Processor
# global variables
from tabfairgdt import NUM_COLS_DTYPES
from tabfairgdt.processor import NAN_KEY
from tabfairgdt.method import CART_FAIR_SPLITTING_METHOD, METHODS_MAP, NA_METHODS, NORMAL_CART
import sys
from tqdm import tqdm


class TABFAIRGDT_FAIR_SPLITTING_CRITERION:
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
                 target_only_fair=True,
                 leaf_relab=False,
                 criterion="dp",
                 re_order=False,
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
        self.leaf_relab = leaf_relab
        self.df_dtypes = dtype_map
        # check init
        self.validator.check_init()
        self.re_order = re_order

        self.standard_method_args = {
            "proper":self.proper, 
            "random_state": self.seed,
            "verbose":verbose
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

    def fit(self, df, dtypes=None, lamda=0.5):
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

        df_cols = list(df.columns)

        self.update_mappings(df)


        df = self.encode(df)







        

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)

        # check processor
        self.validator.check_processor()
        # preprocess
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)


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
            
            if col==self.target and self.leaf_relab:
                self.method[col] = CART_FAIR_SPLITTING_METHOD
                self.method_args_all_cols[col]["criterion"] = self.criterion
            else:
                if col in original_discrete:
                    if (not self.target_only_fair) or (col==self.target):
                        self.method[col] = CART_FAIR_SPLITTING_METHOD
                        self.method_args_all_cols[col]["criterion"] = self.criterion


        self._fit(processed_df)

    def _fit(self, df):
        self.saved_methods = {}

        protected_arrays = df[[self.protected_attribute]].to_numpy().astype(float)


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
            

            if self.method[col] == CART_FAIR_SPLITTING_METHOD or self.method[col] == CART_FAIR_SPLITTING_METHOD:
                col_predictors = col_predictors[col_predictors != self.protected_attribute]

                col_method.fit(X_df=df[col_predictors], y_df=df[col], s=protected_arrays, alpha=self.lamda)
            else:
                col_method.fit(X_df=df[col_predictors], y_df=df[col], alpha=self.lamda)

            # save the method
            self.saved_methods[col] = col_method

    def generate(self, k=None):
        self.k = k

        # check generate
        self.validator.check_generate()
        # generate
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df)

        processed_synth_df = self.decode(processed_synth_df)


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

            if self.method[col] == "sample" and self.re_order:
                synth_df[col] = col_method.predict(synth_df[col_predictors], re_order=self.re_order)
            elif self.method[col] == CART_FAIR_SPLITTING_METHOD or self.method[col] == CART_FAIR_SPLITTING_METHOD:
                col_predictors = col_predictors[col_predictors != self.protected_attribute]
                synth_df[col] = col_method.predict(synth_df[col_predictors])
            else:
                synth_df[col] = col_method.predict(synth_df[col_predictors])
            # profiler.print()
            # change all missing values to 0
            if col in self.processor.processing_dict[NAN_KEY] and self.df_dtypes[col] in NUM_COLS_DTYPES and self.method[col] in NA_METHODS:
                nan_indices = synth_df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                synth_df.loc[nan_indices, col] = 0

            # map dtype to original dtype (only excpetion if column is full of NaNs)
            if synth_df[col].notna().any():
                synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

        return synth_df
    
