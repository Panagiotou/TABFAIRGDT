from __future__ import print_function, division

import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tqdm import tqdm

# code from https://github.com/joymallyac/Fair-SMOTE/blob/master/Generate_Samples.py


class FSMOTE():

    def __init__(self,
                seed=None,
                verbose=False,
                protected_attribute=None,
                target=None,
                target_class_desired=None,
                dtype_map=None,
                ):
        self.seed = seed
        self.verbose = verbose
        self.protected_attribute = protected_attribute
        self.target_class = target
        # self.target_class_desired = int(target_class_desired)
        # assert self.target_class_desired is not None, "Error: target_class_desired must not be None"

        self.dtype_map = dtype_map

        self.knns = []      # one for each split of the df
        self.df = None      # need to save the dataset to get the neighbors during the generation
        self.df_splits = [] # the splits of the dataset for each combination of (PA, target class). NOTE: we suppose binary features
        self.ohe = None
        self.scaler = None
        self.comb_order = None  # the combinations of (PA, target class) values

        random.seed(self.seed)


    def __prepare_data(self):
        # assert self.target_class_desired in pd.to_numeric(self.df[self.target_class]).unique(), f"Error: target_class_desired={self.target_class_desired} must be in df[self.target_class].unique()"

        self.comb_order = []
        for pa_val in self.df[self.protected_attribute].unique():
            for target_val in self.df[self.target_class].unique():
                self.comb_order.append((pa_val, target_val))

        # one hot encode the categorical columns
        cat_cols = [col for col, dtype in self.dtype_map.items() if dtype == "category"]
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(self.df[cat_cols])
        encoded = pd.DataFrame(np.array(encoded), columns=encoder.get_feature_names_out(cat_cols))
        df_ohe = pd.concat([self.df.drop(cat_cols, axis=1), encoded], axis=1)

        # minmax scaler on the data
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_ohe), columns=df_ohe.columns)

        # save encoder and scaler
        self.ohe = encoder
        self.scaler = scaler

        # splits the dataset according to the combinations of (PA, target class)
        for pa_val, target_val in self.comb_order:
            df_split = df_scaled[(self.df[self.protected_attribute] == pa_val) & (self.df[self.target_class] == target_val)]
            print(f"size df_split for {pa_val}, {target_val}: {len(df_split)}")
            self.df_splits.append(df_split)


    def __inverse_transform(self, df):
        # transform the data back to original form
        # reverse minmax scaler
        df = pd.DataFrame(self.scaler.inverse_transform(df), columns=df.columns)

        # reverse one hot encoding
        decoded = self.ohe.inverse_transform(df[self.ohe.get_feature_names_out()])
        df_decoded = pd.DataFrame(decoded, columns=self.ohe.feature_names_in_)
        full_df = pd.concat([df_decoded, df.drop(self.ohe.get_feature_names_out(), axis=1)], axis=1)
        return full_df

        
    def __get_ngbr(self, idx):
        rand_sample_idx = random.randint(0, self.df_splits[idx].shape[0] - 1)
        parent_candidate = self.df_splits[idx].iloc[rand_sample_idx]
        ngbr = self.knns[idx].kneighbors(parent_candidate.values.reshape(1,-1), 3, return_distance=False)
        candidate_1 = self.df_splits[idx].iloc[ngbr[0][0]]
        candidate_2 = self.df_splits[idx].iloc[ngbr[0][1]]
        candidate_3 = self.df_splits[idx].iloc[ngbr[0][2]]
        return parent_candidate, candidate_2, candidate_3
    

    # def __select_data_to_augment(self):
    #     # TODO: delete this
    #     # get the value that is less represented inside the protected attribute
    #     original_len = len(self.df)
    #     value_PA = self.df[self.protected_attribute].value_counts().idxmin()

    #     assert self.target_class_desired in pd.to_numeric(self.df[self.target_class]).unique(), f"Error: target_class_desired={self.target_class_desired} must be in df[self.target_class].unique()"


    #     df = self.df[(self.df[self.protected_attribute] == value_PA) & (pd.to_numeric(self.df[self.target_class]) == self.target_class_desired)]
    #     print("_____________")
    #     print(self.df[self.target_class].value_counts(normalize=True) * 100)
    #     print(f"Chosen target class is {self.target_class}={self.target_class_desired}")

    #     print(self.df[self.protected_attribute].value_counts(normalize=True) * 100)
    #     print(f"Chosen un-privileged protected attribute is {self.protected_attribute}={value_PA}" )
    #     print("_____________")
    #     self.df = df.reset_index(drop=True)

    #     if self.verbose:
    #         print(f"Value to augment from protected attribute: {value_PA}")
    #         print(f"Target value desired for target class {self.target_class}: {self.target_class_desired}")
    #         print(f"Number of samples matching the criteria: {len(self.df)} (original size of dataset: {original_len})")


    def __count_augment_df(self):
        # get the number of examples to augment in each class

        # get most represented combination of (PA, target class)
        comb_values = [0] * len(self.comb_order)
        for i, val in enumerate(self.comb_order):
            pa_val, target_val = val
            comb_values[i] = len(self.df[(self.df[self.protected_attribute] == pa_val) & (self.df[self.target_class] == target_val)])
        max_comb = max(comb_values)

        # get the number of examples to augment in each class
        n_examples_to_augment = [0] * len(self.comb_order)
        for i, comb in enumerate(comb_values):
            n_examples_to_augment[i] = max_comb - comb
        return n_examples_to_augment


    def fit(self, df, n_neighbors=5):
        self.df = df.copy()
        
        # self.__select_data_to_augment()
        self.__prepare_data()
        for df_split in self.df_splits:
            if len(df_split) > 0:
                knn = NN(n_neighbors=n_neighbors, algorithm='auto').fit(df_split.values)    # .values to avoid userwarning during inference
                self.knns.append(knn)
            else:
                self.knns.append(None)


    def generate(self, n_examples_to_augment):      # original name: generate_samples, n_examples_to_augment this argument is not needed, keeping it for overall code consistency


        # get the number of examples to augment for each combination of (PA, target class)
        n_examples_to_augment = self.__count_augment_df()

        # augment for each combination
        for i, n_augment in enumerate(tqdm(n_examples_to_augment, desc="Augmenting subgroups", total=len(n_examples_to_augment), ncols=80)):
            if self.verbose:
                print(f"For protected attribut = {self.comb_order[i][0]} and target class = {self.comb_order[i][1]}")
                print(f"Oringal number of examples: {len(self.df_splits[i])}")
                print(f"Number of examples to augment: {n_augment}")
            if n_augment > 0:
                synthetic_data = []
                for _ in range(n_augment):
                    cr = 0.8
                    f = 0.8
                    parent_candidate, child_candidate_1, child_candidate_2 = self.__get_ngbr(i)
                    new_candidate = []
                    for key, value in parent_candidate.items():
                        if isinstance(parent_candidate[key], bool):
                            new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
                        elif isinstance(parent_candidate[key], str):
                            new_candidate.append(random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
                        elif isinstance(parent_candidate[key], list):
                            temp_lst = []
                            for i, each in enumerate(parent_candidate[key]):
                                temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                                int(parent_candidate[key][i] +
                                                    f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                            new_candidate.append(temp_lst)
                        else:
                            new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))        
                    synthetic_data.append(new_candidate)

                synth_df = pd.DataFrame(synthetic_data, columns=self.df_splits[i].columns)
                synth_df = self.__inverse_transform(synth_df)
                self.df = pd.concat([self.df, synth_df], ignore_index=True)
        return self.df

