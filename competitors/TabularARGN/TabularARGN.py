from mostlyai import engine
from mostlyai.engine._common import ProgressCallback 
from sklearn.preprocessing import KBinsDiscretizer

from tqdm import tqdm

from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd

# https://colab.research.google.com/github/mostly-ai/mostlyai/blob/main/docs/tutorials/fairness/fairness.ipynb#scrollTo=99baee6f076612c
# https://arxiv.org/pdf/2311.03000
# https://arxiv.org/pdf/2501.12012

   
class TabularARGN():
    def __init__(self,
                seed=None,
                verbose=False,
                protected_attribute=None,
                target=None,
                target_class_desired = None,
                protected_attribute_under_represented = None,
                dtype_map=None,
                continuous_protected_attribute=False,
                epochs=100
                ):
        
        self.seed = seed 
        self.verbose = verbose
        self.protected_attribute = protected_attribute
        self.continuous_protected_attribute = continuous_protected_attribute
        self.target_class = target
        self.target_class_desired = str(target_class_desired)
        self.protected_attribute_under_represented = str(protected_attribute_under_represented)

        assert self.target_class_desired is not None, "Error: target_class_desired must not be None"
        assert self.protected_attribute_under_represented is not None, "Error: protected_attribute_under_represented must not be None"

        self.dtype_map = dtype_map

        # self.device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

        ws_base = Path("ws_mostly")
        ws_base.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=ws_base))

        
        self.fairness_config = {
            "target_column": self.target_class,  # define fairness target
            "sensitive_columns": [self.protected_attribute],  # define sensitive columns
        }
        self.max_epochs = epochs


    def generate(self, size_of_fake_data):
          
        engine.generate(workspace_dir=self.temp_dir, fairness=self.fairness_config, lamda = self.lamda)     # use model to generate synthetic samples to `{self.temp_dir}/SyntheticData`
        fair_syn = pd.read_parquet(self.temp_dir / "SyntheticData") # load synthetic data
        shutil.rmtree(self.temp_dir)

        if self.continuous_protected_attribute:
            fair_syn.drop([self.protected_attribute + "_binned"], axis="columns", inplace=True)
        fair_syn = fair_syn.astype(self.dtype_map)
        return fair_syn
    


    def fit(self, df, lamda=0.5):

        df[self.target_class] = df[self.target_class].astype('category')

        if self.continuous_protected_attribute:
            discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
            protected_binned = np.array(discretizer.fit_transform(df[self.protected_attribute].values.reshape(-1, 1)))

            # protected_binned_med = np.vectorize(lambda x: bin_medians[0][int(x)])(protected_binned)

            df[self.protected_attribute + "_binned"] = protected_binned
            df[self.protected_attribute + "_binned"] = df[self.protected_attribute + "_binned"].astype(int).astype('category')
            self.fairness_config["sensitive_columns"] = [self.protected_attribute + "_binned"]

        engine.split(                         # split data as PQT files for `trn` + `val` to `{ws}/OriginalData/tgt-data`
        workspace_dir=self.temp_dir,
        tgt_data=df,
        model_type="TABULAR",
        )

        self.lamda = lamda

        engine.analyze(workspace_dir=self.temp_dir)      # generate column-level statistics to `{self.temp_dir}/ModelStore/(tgt|ctx)-data/stats.json`
        engine.encode(workspace_dir=self.temp_dir)       # encode training data to `{self.temp_dir}/OriginalData/encoded-data`
        engine.train(                         # train model and store to `{self.temp_dir}/ModelStore/model-data`
            workspace_dir=self.temp_dir,
            # update_progress=TrainingProgress(max_epochs=self.max_epochs),
            max_epochs=self.max_epochs
        )




