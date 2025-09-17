# Evaluate the quality of the generated data

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from synthcity.metrics.eval_statistical import PRDCScore
from synthcity.plugins.core.dataloader import GenericDataLoader
from sdmetrics.single_column import KSComplement, TVComplement
import lightgbm as lgb
import sys
import os 
import contextlib


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# machine learning utility
## do we need this? eval done on the fair data is very similar


# def get_data_for_eval(real: pd.DataFrame,
#                       fake: pd.DataFrame,
#                       protected_attribute: str):
#     """
#     Split the data according to the protected attribute.
#     Example: if PA is sex and the values are male and female, returns 4 dataframes: male_real, female_real, male_fake, female_fake.
#     """
#     uniques = real[protected_attribute].unique().tolist()
#     if len(uniques) != 2:
#         raise ValueError("Protected attribute must have 2 unique values.")
    
#     df_real_0 = real[real[protected_attribute] == uniques[0]]
#     df_real_1 = real[real[protected_attribute] == uniques[1]]
#     df_fake_0 = fake[fake[protected_attribute] == uniques[0]]
#     df_fake_1 = fake[fake[protected_attribute] == uniques[1]]

#     return df_real_0, df_real_1, df_fake_0, df_fake_1
    

# detection score
def detection_score(real: pd.DataFrame,
                    fake: pd.DataFrame,
                    # model: BaseEstimator,
                    # categorical_columns: list[str],
                    n_folds: int = 3,
                    scoring: str = "roc_auc",
                    random_state: int = 42) -> tuple[float, float]:
    """
    Computes the detection score of the synthetic data according to the scoring and the number of folds given.
    Detection score is how well the synthetic data can be distinguished from the real data.

    Inputs:
    - real: pd.DataFrame, the real data
    - fake: pd.DataFrame, the fake, generated data
    - model: BaseEstimator, the model to use for detection
    - categorical_columns: list[str], the categorical columns in the data
    - n_folds: int, the number of folds for cross-validation. Default = 5.
    - scoring: str, the scoring method to use. Default = "roc_auc".
    - random_state: int, the random state to use. Default = 42.

    Returns the mean detection score and its standard deviation.
    """
    # equalize the number of samples
    if len(real) > len(fake):
        real = real.sample(n=len(fake), random_state=random_state)
    else:
        fake = fake.sample(n=len(real), random_state=random_state)





    categorical_columns = list(real.select_dtypes(include=['object', 'category']).columns)
    print("detection_score, selected Cat Cols", categorical_columns)


    real = real.astype({col: 'str' for col in real.select_dtypes('category').columns})
    fake = fake.astype({col: 'str' for col in fake.select_dtypes('category').columns})

    
    if real.isna().any().any() or fake.isna().any().any():
        print("real or fake has Nan")
        exit(1)

    # add labels and concat the data
    real["fake"] = 0
    fake["fake"] = 1
    data = pd.concat([real, fake], ignore_index=True)

    model = lgb.LGBMClassifier(categorical_feature=categorical_columns)

    # cross val
    y = data["fake"]
    x = data.drop(columns=["fake"])

    x[categorical_columns] = x[categorical_columns].astype('category')    

    with suppress_stdout():
        scores = cross_val_score(model, x, y, cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state), scoring=scoring)

    return [scores.mean()]#, scores.std()


# PRDC (precision, recall, density, coverage)
def prdc_score(real: pd.DataFrame,
               fake: pd.DataFrame,
            #    categorical_columns: list[str],
               neares_k: int = 5) -> tuple[float, float, float, float]:
    """
    Computes the PRDC (precision, recall, density, coverage) score of the synthetic data.
    
    Inputs:
    - real: pd.DataFrame, the real data
    - fake: pd.DataFrame, the fake, generated data
    - categorical_columns: list[str], the categorical columns in the data
    - nearest_k: int, the number of nearest neighbors to use. Default = 5.

    Returns the prdc values (in this order)
    """

    if len(fake) > len(real):
        return [0, 0, 0, 0]
    
    if len(fake) > 20000:   
        print("subsampling for prdc score")
        fake = fake.sample(n=20000)  # Ensures reproducibility
        real = real.sample(n=20000)  # Ensures reproducibility


    categorical_columns = list(real.select_dtypes(include=['object', 'category']).columns)
    print("prdc_score, selected Cat Cols", categorical_columns)


    
    real = real.astype({col: 'str' for col in real.select_dtypes('category').columns})
    fake = fake.astype({col: 'str' for col in fake.select_dtypes('category').columns})

    # # one hot dataset
    combined = pd.concat([real, fake], ignore_index=True)

    # Apply get_dummies to the concatenated dataframe
    combined_dummies = pd.get_dummies(combined, columns=categorical_columns)

    # Split the combined dataframe back into real and fake datasets
    real_dummies = combined_dummies.iloc[:len(real)]
    fake_dummies = combined_dummies.iloc[len(real):]

    # normalize between 0 and 1 (prdc uses knn)
    scaler = MinMaxScaler()
    real = scaler.fit_transform(real_dummies)
    fake = scaler.transform(fake_dummies)

    # combined[combined.select_dtypes(include=['object', 'category']).columns] = combined.select_dtypes(include=['object', 'category']).apply(lambda x: x.astype('category').cat.codes)
    # real_dummies = combined.iloc[:len(real)]
    # fake_dummies = combined.iloc[len(real):] # This is not one-hot !!!!

    real = scaler.fit_transform(real_dummies)
    fake = scaler.transform(fake_dummies)
    # compute prdc using synthcity
    real = GenericDataLoader(real)
    fake = GenericDataLoader(fake)

    # print("PRDCScore")
    prdc = PRDCScore(nearest_k=neares_k)
    # print("Done")
    scores = prdc.evaluate(real, fake)
    # print("Eval Done")

    return [scores["precision"], scores["recall"], scores["density"], scores["coverage"]]


def ks_score(real: pd.DataFrame,
             fake: pd.DataFrame):
            #  continous_columns: list[str]):
    """
    Computes the Kolmogorov-Smirnov score of the synthetic data for the continuous columns.
    Score is between [0, 1], 1 = distributions are similar, 0 = distributions are different.

    Computes the score for each continuous column of the dataset. Returns the mean and a dictionary with value for each continuous column.
    
    Inputs:
    - real: pd.DataFrame, the real data
    - fake: pd.DataFrame, the fake, generated data
    - continous_columns: list[str], the continuous columns in the data
    """
    all_scores = {}
    mean_score = 0
    continous_columns = real.select_dtypes(exclude=['object', 'category']).columns
    print("KS-score, selected Cont Cols", continous_columns)

    for col in continous_columns:
        score = KSComplement.compute(real_data=real[col], synthetic_data=fake[col])
        all_scores[col] = score
        mean_score += score
    if len(continous_columns) == 0: return [0]
    mean_score /= len(continous_columns)
    return [mean_score]#, all_scores
    

def tv_score(real: pd.DataFrame,
             fake: pd.DataFrame):
            #  categorical_columns: list[str]):
    """
    Computes the Total Variation score of the synthetic data for the categorical columns.
    Score is between [0, 1], 1 = distributions are similar, 0 = distributions are different.

    Computes the score for each categorical column of the dataset. Returns the mean and a dictionary with value for each categorical column.

    Inputs:
    - real: pd.DataFrame, the real data
    - fake: pd.DataFrame, the fake, generated data
    - categorical_columns: list[str], the categorical columns in the data
    """
    all_scores = {}
    mean_score = 0

    categorical_columns = list(real.select_dtypes(include=['object', 'category']).columns)
    print("TV-score, selected Cat Cols", categorical_columns)

    for col in categorical_columns:
        score = TVComplement.compute(real_data=real[col], synthetic_data=fake[col])
        all_scores[col] = score
        mean_score += score
    mean_score /= len(categorical_columns)
    return [mean_score]#, all_scores



