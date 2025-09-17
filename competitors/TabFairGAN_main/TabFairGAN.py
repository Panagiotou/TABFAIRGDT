import torch
import torch.nn.functional as f
from torch import nn
import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import argparse

from tqdm import tqdm

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
with_fairness = subparser.add_parser('with_fairness')

with_fairness.add_argument("df_name", help="Reference dataframe", type=str)
with_fairness.add_argument("Y", help="Label (decision)", type=str)
with_fairness.add_argument("underprivileged_value", help="Value for underpriviledged group", type=str)
with_fairness.add_argument("desirable_value", help="Desired label (decision)", type=str)
with_fairness.add_argument("num_epochs", help="Total number of epochs", type=int)
with_fairness.add_argument("batch_size", help="the batch size", type=int)
with_fairness.add_argument("num_fair_epochs", help="number of fair training epochs", type=int)
with_fairness.add_argument("lambda_val", help="lambda parameter", type=float)
with_fairness.add_argument("fake_name", help="name of the produced csv file", type=str)
with_fairness.add_argument("size_of_fake_data", help="how many data records to generate", type=int)




class TabFairGAN():
    def __init__(self,
                seed=None,
                verbose=False,
                protected_attribute=None,
                target=None,
                target_class_desired = None,
                protected_attribute_under_represented = None,
                dtype_map=None,
                ):
        
        self.seed = seed 
        self.verbose = verbose
        self.protected_attribute = protected_attribute

        self.target_class = target
        self.target_class_desired = str(target_class_desired)
        self.protected_attribute_under_represented = str(protected_attribute_under_represented)

        assert self.target_class_desired is not None, "Error: target_class_desired must not be None"
        assert self.protected_attribute_under_represented is not None, "Error: protected_attribute_under_represented must not be None"

        self.dtype_map = dtype_map

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")


        # S = protected_attributes
        # Y = Y
        # S_under = underprivileged_value
        # Y_desire = desirable_value

        # df = pd.read_csv(df_name)

        # df[S] = df[S].astype(object)
        # df[Y] = df[Y].astype(object)

    def generate(self,size_of_fake_data):

        fake_numpy_array = self.generator(torch.randn(size=(size_of_fake_data, self.input_dim), device=self.device)).cpu().detach().numpy()
        fake_df = self.get_original_data(fake_numpy_array, self.ohe, self.scaler)
        fake_df = fake_df[self.columns]
        return fake_df
    
    def get_ohe_data(self, df):
        df_int = df.select_dtypes(['float', 'integer']).values
        continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
        ##############################################################


        if len(continuous_columns_list) > 0:
            scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
            df_int = scaler.fit_transform(df_int)
            numerical_array = df_int
        else:
            numerical_array = None
            scaler = None

        df_cat = df.select_dtypes('object')
        df_cat_names = list(df.select_dtypes('object').columns)

        ohe = OneHotEncoder(sparse_output=False)
        ohe_array = np.array(ohe.fit_transform(df_cat))

        cat_lens = [i.shape[0] for i in ohe.categories_]
        discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

        S_start_index = len(continuous_columns_list) + sum(
            list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(self.protected_attribute)])
        Y_start_index = len(continuous_columns_list) + sum(
            list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(self.target_class)])

        if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(self.protected_attribute)][0] == self.protected_attribute_under_represented:
            underpriv_index = 0
            priv_index = 1
        else:
            underpriv_index = 1
            priv_index = 0

        if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(self.target_class)][0] == self.target_class_desired:
            desire_index = 0
            undesire_index = 1
        else:
            desire_index = 1
            undesire_index = 0


        print("_____________")
        print(df[self.target_class].value_counts(normalize=True) * 100)
        print(f"Chosen desired index = {desire_index} which corresponds to value {self.target_class}={ohe.categories_[list(discrete_columns_ordereddict.keys()).index(self.target_class)][desire_index]}" )

        print(df[self.protected_attribute].value_counts(normalize=True) * 100)
        print(f"Chosen un-privileged protected attribute underpriv_index = {underpriv_index} which corresponds to value {self.protected_attribute}={ohe.categories_[list(discrete_columns_ordereddict.keys()).index(self.protected_attribute)][underpriv_index]}" )
        print("_____________")

        if numerical_array is not None:
            final_array = np.hstack((numerical_array, ohe_array))
        else:
            final_array = ohe_array

        return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index


    def get_original_data(self, df_transformed, ohe, scaler):
        if scaler is not None:
            df_ohe_int = df_transformed[:, :self.original_numeric.shape[1]]
            df_ohe_int = scaler.inverse_transform(df_ohe_int)
            df_int = pd.DataFrame(df_ohe_int, columns=self.original_numeric.columns)
        else:
            df_int = None

        df_ohe_cats = df_transformed[:, self.original_numeric.shape[1]:]
        df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
        df_cat = pd.DataFrame(df_ohe_cats, columns=self.original_discrete.columns)
        if df_int is not None:
            return pd.concat([df_int, df_cat], axis=1)
        else:
            return df_cat


    def prepare_data(self, df, batch_size):
        ohe, scaler, discrete_columns, continuous_columns, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = self.get_ohe_data(df)
        input_dim = df_transformed.shape[1]

        X_train = df_transformed
        data_train = X_train.copy()

        data = torch.from_numpy(data_train).float()


        train_ds = TensorDataset(data)
        train_dl = DataLoader(train_ds, batch_size = batch_size, drop_last=True)
        return ohe, scaler, input_dim, discrete_columns, continuous_columns ,train_dl, data_train, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index


    def get_gradient(self, crit, real, fake, epsilon):
        mixed_data = real * epsilon + fake * (1 - epsilon)

        mixed_scores = crit(mixed_data)

        gradient = torch.autograd.grad(
            inputs=mixed_data,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient


    def gradient_penalty(self, gradient):
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)

        penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty


    def get_gen_loss(self, crit_fake_pred):
        gen_loss = -1. * torch.mean(crit_fake_pred)

        return gen_loss


    def get_crit_loss(self, crit_fake_pred, crit_real_pred, gp, c_lambda):
        crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

        return crit_loss

    def fit(self, df, epochs=200, batch_size=256, fair_epochs=30, lamda=0.5):


        for column, dtype in self.dtype_map.items():
            if dtype == "category":
                df[column] = df[column].astype("object")
            else:
                df[column] = df[column].astype("int")
    
        df[self.target_class] = df[self.target_class].astype("object")


        assert self.target_class_desired in df[self.target_class].astype(str).unique(), f"Error: target_class_desired={self.target_class_desired} must be in df[self.target_class].unique()"
        assert self.protected_attribute_under_represented in df[self.protected_attribute].astype(str).unique(), f"Error: protected_attribute_under_represented={self.protected_attribute_under_represented} must be in df[self.target_class].unique()"


        self.original_numeric =  df.select_dtypes(['float', 'integer'])
        self.original_discrete = df.select_dtypes('object')



        if self.verbose:
            print("Numeric cols", list(self.original_numeric.columns))
            print("Discrete cols", list(self.original_discrete.columns))


        self.columns = df.columns

        ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = self.prepare_data(df, batch_size)


        self.input_dim = data_train.shape[-1]


        generator = Generator(input_dim, continuous_columns, discrete_columns).to(self.device)
        critic = Critic(input_dim).to(self.device)

        second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(self.device)

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # loss = nn.BCELoss()
        critic_losses = []
        cur_step = 0

        for i in tqdm(range(epochs), desc="Training epochs TabFairGAN", position=0, leave=True):
            # j = 0
            ############################
            # if i + 1 <= (epochs - fair_epochs):
            #     print("training for accuracy")
            # if i == (epochs - fair_epochs):
            #     print("\t\ttraining for fairness")
            for data in train_dl:
                data[0] = data[0].to(self.device)
                crit_repeat = 4
                mean_iteration_critic_loss = 0
                for k in range(crit_repeat):
                    # training the critic
                    crit_optimizer.zero_grad()
                    fake_noise = torch.randn(size=(batch_size, input_dim), device=self.device).float()
                    fake = generator(fake_noise)

                    crit_fake_pred = critic(fake.detach())
                    crit_real_pred = critic(data[0])

                    epsilon = torch.rand(batch_size, input_dim, device=self.device, requires_grad=True)
                    gradient = self.get_gradient(critic, data[0], fake.detach(), epsilon)
                    gp = self.gradient_penalty(gradient)

                    crit_loss = self.get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                    mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                    crit_loss.backward(retain_graph=True)
                    crit_optimizer.step()
                #############################
                if cur_step > 50:
                    critic_losses += [mean_iteration_critic_loss]

                #############################
                if i + 1 <= (epochs - fair_epochs):
                    # training the generator for accuracy
                    gen_optimizer.zero_grad()
                    fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=self.device).float()
                    fake_2 = generator(fake_noise_2)
                    crit_fake_pred = critic(fake_2)

                    gen_loss = self.get_gen_loss(crit_fake_pred)
                    gen_loss.backward()

                    # Update the weights
                    gen_optimizer.step()

                ###############################
                if i + 1 > (epochs - fair_epochs):
                    # training the generator for fairness
                    gen_optimizer_fair.zero_grad()
                    fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=self.device).float()
                    fake_2 = generator(fake_noise_2)

                    crit_fake_pred = critic(fake_2)

                    gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
                    gen_fair_loss.backward()
                    gen_optimizer_fair.step()
                cur_step += 1

        self.generator = generator
        self.critic = critic
        self.ohe = ohe
        self.scaler = scaler


class Generator(nn.Module):
    def __init__(self, input_dim, continuous_columns, discrete_columns):
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._discrete_columns = discrete_columns
        self._num_continuous_columns = len(continuous_columns)

        self.lin1 = nn.Linear(self._input_dim, self._input_dim)
        self.lin_numerical = nn.Linear(self._input_dim, self._num_continuous_columns)

        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._input_dim, value)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        # x = f.leaky_relu(self.lin1(x))
        # x_numerical = f.leaky_relu(self.lin_numerical(x))
        x_numerical = f.relu(self.lin_numerical(x))
        x_cat = []
        for key in self.lin_cat:
            x_cat.append(f.gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self._input_dim = input_dim
        # self.dense1 = nn.Linear(109, 256)
        self.dense1 = nn.Linear(self._input_dim, self._input_dim)
        self.dense2 = nn.Linear(self._input_dim, self._input_dim)
        # self.dense3 = nn.Linear(256, 1)
        # self.drop = nn.Dropout(p=0.2)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = f.leaky_relu(self.dense1(x))
        # x = self.drop(x)
        # x = f.leaky_relu(self.dense2(x))
        x = f.leaky_relu(self.dense2(x))
        # x = self.drop(x)
        return x


class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        # print(x[0,64])
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        # disp = (torch.mean(G[:,1]*I[:,1])/(x[:,65].sum())) - (torch.mean(G[:,0]*I[:,0])/(x[:,64].sum()))
        # disp = -1.0 * torch.tanh(torch.mean(G[:,0]*I[:,1])/(x[:,64].sum()) - torch.mean(G[:,1]*I[:,1])/(x[:,65].sum()))
        # gen_loss = -1.0 * torch.mean(crit_fake_pred)
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index]) / (
            x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(
            G[:, self._priv_index] * I[:, self._desire_index]) / (
                                x[:, self._S_start_index + self._priv_index].sum())) - 1.0 * torch.mean(
            crit_fake_pred)
        # print(disp)
        return disp




