import os
import pandas as pd
import json
from ucimlrepo import fetch_ucirepo
from folktables import ACSDataSource, ACSIncome
from folktables import ACSDataSource, ACSIncome

class Dataset():
    def __init__(self, dataset_name, binary_features=False, ignore_features=[], protected_attribute="sex"):
        self.binary_features = binary_features
        self.ignore_features = ignore_features
        self.protected_attribute = protected_attribute
        self.load_dataset(dataset_name)
        self.dataset_name = dataset_name


    def encode(self, df):
        df_encoded = df.copy()
        for column in df_encoded.columns:
            dtype = self.dtype_map[column]
            if dtype == "category":
                df_encoded[column] = df_encoded[column].map(self.reverse_mappings[column]).astype("category")
        return df_encoded

    def decode(self, df, keep_dtypes=False):
        df_decoded = df.copy()
        for column, mapping in self.original_mappings.items():
            if column in df_decoded.columns:
                df_decoded[column] = df_decoded[column].map(mapping)
        if keep_dtypes:
            df_decoded = df_decoded.astype(df.dtypes)
        return df_decoded

    def acsi_get_occupation_category(self, code):
        """
        Maps occupation codes to their respective categories.
        
        Args:
            code (int): Occupation code
        Returns:
            str: Category name
        """
        code = int(code)
        
        if 0 <= code <= 3550:
            return "Management_Business_Science_and_Arts"
        elif 3601 <= code <= 4655:
            return "Service"
        elif 4700 <= code <= 5940:
            return "Sales_and_Office"
        elif 6005 <= code <= 7640:
            return "Natural_Resources_Construction_and_Maintenance"
        elif 7700 <= code <= 9760:
            return "Production_Transportation_and_Material_Moving"
        else:
            return "Unknown"
        
    def acsi_get_native_country_category(self, code):

        code = int(code)
        
        if 0 <= code <= 56:
            return "US"
        else:
            return "non-US"
    

    def load_dataset(self, dataset_name):

        info_path = f'tabular_datasets/Info/{dataset_name}_info.json'

        if not os.path.isfile(info_path):
            raise NotImplementedError(f"Dataset {dataset_name} is not defined, create the file {info_path}.")

        with open(info_path, 'r') as f:
            info = json.load(f)


        self.ignore_features = info["ignore_features"]
        url = info["url"]

        self.target = info["target"]
        self.favorable_target = info["favorable_target"]
        self.protected_attribute_under_represented = info["protected_attribute_under_represented"]

        folder_path = "tabular_datasets/{}".format(dataset_name)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

        print(json_file_path)

        # Check if the JSON file exists locally
        if not os.path.isfile(json_file_path):
            # File does not exist locally, download the CSV dataset
            if "archive.ics.uci.edu" in url:
                # fetch the dataset from UCI repository
                try:
                    idx = int(url.split("/")[4])  # expect url to be formatted as https://archive.ics.uci.edu/dataset/idx/name
                except ValueError:
                    raise ValueError(f"Invalid UCI repository URL. Expected format: https://archive.ics.uci.edu/dataset/idx/name, given url is {url}")
                dataset_uci = fetch_ucirepo(id=idx)
                df = dataset_uci.data["original"]
            elif "folktables" in url:
                survey_year = info["survey_year"]
                survey = info["survey"]
                horizon = info["horizon"]
                state = info["state"]

                data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)
                state_data = data_source.get_data(states=[state], download=True)

                if "RELP" not in state_data.columns: # I dont know why this happens
                    state_data["RELP"] = 0

                df, labels, _ = ACSIncome.df_to_pandas(state_data)

                df.drop("RELP", axis="columns", inplace=True)

                df[info["target"]] = labels.astype(int)
    
                df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"})
                df['OCCP'] = df['OCCP'].apply(self.acsi_get_occupation_category)
                df['POBP'] = df['POBP'].apply(self.acsi_get_native_country_category)

                # for col in df.columns:
                #     unique_counts = df[col].nunique()
                #     unique_values = df[col].unique()
                #     value_counts = df[col].value_counts(normalize=True) * 100  # Calculate percentages
                #     sorted_percentages = value_counts.sort_values(ascending=False)  # Sort in descending order

                #     print(f"\t{col}: {unique_counts} unique values")
                #     print("Unique values:", unique_values)
                #     print("Sorted percentages:\n", sorted_percentages, "\n")
                # exit(1)
            else:
                df = pd.read_csv(url)

            if "bank" in url:
                df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)  # convert target to binary
        
        
            print(df.columns)
            df = df.drop(self.ignore_features, axis="columns")

            if dataset_name == "census":
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].str.strip()   # dataset is loaded with extra spaces in the strings
                df["income"] = df["income"].apply(lambda x: 1 if x == "50000+." else 0) # convert income to binary

            if dataset_name == "credit":
                df["X2"] = df["X2"].map({1: "male", 2: "female"})

            if dataset_name == "bar":
                df["male"] = df["male"].map({1: "male", 0: "female"})

            if dataset_name == "portuguese":
                df["G3"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)  # convert final grade to binary

            if dataset_name == "bank":
                df["marital"] = df["marital"].replace({"divorced": "single"})
                df["y"] = df["y"].map({"yes": 1, "no": 0})  # convert target to binary

            if dataset_name == "diabetes":
                df["readmitted"] = df["readmitted"].map({">30": 0, "<30": 1})
                df["age"] = df["age"].apply(lambda x: x.replace("[", "("))  # to this to prevent error with lgb and json file
            
            if dataset_name == "recid" or dataset_name == "violent":
                for col in df.columns:
                    for v in df[col].unique():
                        if "," in str(v):
                            df[col] = df[col].apply(lambda x: str(x).replace(",", "."))  # replace commas with dots in all columns
                        if ":" in str(v):
                            df[col] = df[col].apply(lambda x: str(x).replace(":", "."))

            num_rows_with_nan = df.isnull().sum(axis=1).gt(0).sum()
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"Number of rows containing NaN values {num_rows_with_nan}, dropping them. Remained with {len(df)} rows.")
            if df.isnull().values.any() or df.empty:
                print("Nan in dataset")
                exit(1)

            df.to_json(json_file_path, orient='records', lines=True)
        else:
            # JSON file already exists locally, load it directly
            df = pd.read_json(json_file_path, orient='records', lines=True)

        if "rename" in info:
            df.rename(columns=info["rename"], inplace=True)

        self.dtype_map = info["dtype_map"]

        self.column_names = list(self.dtype_map.keys())

        self.original_mappings = {}
        for column, dtype in self.dtype_map.items():
            if dtype == "category":
                self.original_mappings[column] = dict(enumerate(df[column].astype("category").cat.categories))

        self.reverse_mappings = {col: {v: k for k, v in self.original_mappings[col].items()} for col in self.original_mappings}


        for column, dtype in self.dtype_map.items():
            if dtype == "category":
                df[column] = df[column].astype("object")


        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).astype("category")

        self.original_dataframe = df.copy()
        self.original_dataframe_encoded = self.encode(df).copy()

        df_dummy_drop = df.copy()
        df_dummy_drop = df_dummy_drop.drop(columns=[self.target, self.protected_attribute])

        self.categorical_input_cols = [col for col in df_dummy_drop.columns if self.dtype_map[col] == "category"]

        self.continuous_input_cols = [col for col in df_dummy_drop if col not in self.categorical_input_cols]

        self.categorical_input_col_locations = [df_dummy_drop.columns.get_loc(col) for col in self.categorical_input_cols]

        print("Dataset {} has {} categorical and {} numerical columns.".format(dataset_name, self.categorical_input_cols, self.continuous_input_cols))
        return self







    ## process data ACSI
    # df = pd.read_excel("tabular_datasets/acsi/ACSI Data 2015.xlsx")  # from https://data.mendeley.com/tabular_datasets/64xkbj2ry5/1
    # cs = []
    # for col in df.columns:
    #     count = df[col].isna().sum()
    #     if count > 1000:
    #         cs.append(col)
    # print(cs)
    # df = df.drop(columns=cs)
    # print(df.shape)
    # df = df.dropna()
    # print(df.shape)
    # df = df.drop_duplicates()
    # df["GENDER"] = df["GENDER"].apply(lambda x: "Male" if x == 1.0 else "Female")
    # df["INCOME"] = df["INCOME"].apply(lambda x: 1 if x >= 5 else 0)

    # for col in df.columns:
    #     print(f'\t"{col}": "int",')

    # df.to_json("tabular_datasets/acsi/acsi.json", orient="records", lines=True)
