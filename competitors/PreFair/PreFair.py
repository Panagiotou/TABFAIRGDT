

# how to determine which attribue is admissible?
#   paper considers 3 sets: protected, admissible, and outcome
#      protected: These are considered sensitive and unactionable, such as race and gender
#      admissible: These are considered actionable for decision-making, despite their potential causal links to protected attributes
#      outcome: This is the target variable or variables to predict given the other attributes
#   they are disjoint sets but do NOT cover the full set of attributes

# not sure of the format of the data: seems to encode all the features, even the numerical ones

# repository contains two codes: the datasynthesizer and private-pgm
# the main code for prefair is private-pgm, datasynthesizer is used for the privBayes mentioned in the paper


# mbi folder from original repo https://github.com/ryan112358/private-pgm/tree/aae58df3dc27b9d7ceb9eeab75a02549b3bc870e/src/mbi

import pandas as pd
import itertools
import numpy as np
# from competitors.PreFair.mbi import FactoredInference, Dataset, Domain
from competitors.PreFair.mbi.dataset import Dataset
import competitors.PreFair.mst_fair_greedy as fairMST
# import competitors.PreFair.mst_fair_optimal as fairMSTOpt

# from disjoint_set import DisjointSet
# from cdp2adp import cdp_rho
# from scipy.special import logsumexp
# import networkx as nx
# import scipy
# import argparse
# import heapq
# import timeit


class PreFair():

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
        if type(self.target_class) != list:
            self.target_class = [self.target_class]
        self.target_class_desired = int(target_class_desired)
        self.dtype_map = dtype_map
        self.categorical_feat = [col for col, dtype in self.dtype_map.items() if dtype == "category"]  # suppose outcome is already binarized

        # transform PA to list if not already the case
        if type(self.protected_attribute) != list:
            self.protected_attribute = [self.protected_attribute]


    def __transform_dataset(self):
        self.mapping = {}
        for col in self.df.columns:
            self.mapping[col] = {val: i for i, val in enumerate(self.df[col].unique())}
            self.df[col] = self.df[col].map(self.mapping[col])

    
    def __detransform_dataset(self, data):
        def reverse_mapping(d):
            return {v: k for k, v in d.items()}
        for col in data.columns:
            data[col] = data[col].map(reverse_mapping(self.mapping[col]))
        return data


    def __compute_df_domain(self):
        self.domain = {}
        for col in self.df.columns:
            self.domain[col] = len(self.df[col].unique())

    
    def fit(self,
            data: pd.DataFrame,
            # admissible_features: list[str],
            epsilon: float = 0.1,
            delta: float = 1e-9,
            degree: int = 2,
            num_marginals = None,
            max_cells: int = 100000):
        self.df = data.copy()
        self.admissible_features = [x for x in list(data.columns) if x not in self.target_class + self.protected_attribute]

        print("Admissible features", self.admissible_features)
        self.epsilon = epsilon
        self.delta = delta
        self.degree = degree
        self.num_marginals = num_marginals
        self.max_cells = max_cells

        self.__compute_df_domain()
        self.__transform_dataset()

        self.dataset = Dataset.load(self.df, self.domain)
        # self.dataset = Dataset.load("competitors/PreFair/adult.csv", "competitors/PreFair/adult-domain.json")
        
        workload = list(itertools.combinations(self.dataset.domain, self.degree))
        workload = [cl for cl in workload if self.dataset.domain.size(cl) <= self.max_cells]
        if self.num_marginals is not None:
            prng = np.random.default_rng(self.seed)
            workload = [workload[i] for i in prng.choice(len(workload), num_marginals, replace=False)]



    
    def generate(self, n_samples=-1):
        synth = fairMST.MST(self.dataset, self.epsilon, self.delta, self.target_class, self.admissible_features)
        # synth = fairMST.MST(self.dataset, 0.1, 1e-9, ["income>50K"], ['workclass','fnlwgt','education-num','occupation','capital-gain','capital-loss','hours-per-week'])
        synth = self.__detransform_dataset(synth.df)
        return synth






