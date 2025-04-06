import numpy as np
import uproot as ur # type: ignore
import pandas as pd # type: ignore
import pickle # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import torch
import os
from torch.utils.data import Dataset, DataLoader

class Transform:
    def __call__(self, data: np.ndarray):
        return data

class Normalize(Transform):
    def __call__(self, data: np.ndarray):
        data = data.astype(np.float64)

        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)

        data -= min_values
        data /= (max_values - min_values)

        return data
    
class ParticlesDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform: Transform = Transform()):
        if data.shape[0] != labels.shape[0]:
            raise RuntimeError("Training data and training labels have size mismatch:", self.__class__.__name__)
        
        self.data = torch.tensor(transform(data), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return (self.data[index], self.labels[index])
    
    @property
    def classes(self):
        return ['Background', 'Signal']
    
    @property
    def features(self):
        return ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']
    
class DataManipulator:
    def __init__(self, batch_size, transform: Transform = Transform(), seed = 42069):
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed
        self.X = None
        self.y = None
        self.total_df = None
        self.under_sampled_df = None
        self.training_df = None
        self.valid_df = None
        self.test_df = None

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.X_u_train = None
        self.X_u_val = None
        self.X_u_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.y_u_train = None
        self.y_u_val = None
        self.y_u_test = None

    def get_dataloaders(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray):
        

        training_dataset = ParticlesDataset(data=X_train, labels=y_train, transform=self.transform)
        training_dataloader = DataLoader(dataset=training_dataset, batch_size=self.batch_size, shuffle=True)

        validation_dataset = ParticlesDataset(data=X_val, labels=y_val, transform=self.transform)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=self.batch_size, shuffle=False)

        test_dataset = ParticlesDataset(data=X_test, labels=y_test, transform=self.transform)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        return (training_dataloader, validation_dataloader, test_dataloader)
    
    def load_all_data(self, pickled_X:str, pickled_y:str, uproot_background:str, uproot_signal:str):
        if os.path.exists(pickled_X) and os.path.exists(pickled_y):
            with open(pickled_X, 'rb') as f:
                X = pickle.load(f)
            with open(pickled_y, 'rb') as f:
                y = pickle.load(f)
            self.X = X
            self.y = y
            return (X, y)

        background_data_file = ur.open(uproot_background)
        signal_data_file = ur.open(uproot_signal)
        features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']
        preselection = '(KLmassD0 > 2.) & ((Mll>1.05) & (Mll<2.45))'

        sig_dict = signal_data_file['mytree'].arrays(features, library='np', cut=preselection)
        bkg_dict = background_data_file['mytree'].arrays(features, library='np', cut=preselection)
        backgr = np.stack(list(bkg_dict.values()))
        signal = np.stack(list(sig_dict.values()))

        X = np.transpose(np.concatenate((signal, backgr), axis=1))
        y = np.concatenate((np.ones(signal.shape[1]), np.zeros(backgr.shape[1])))

        with open(pickled_X, 'wb') as f:
            pickle.dump(X, f)
        with open(pickled_y, 'wb') as f:
            pickle.dump(y, f)
        
        self.X = X
        self.y = y
        return (X, y)
    
    def plot_event_distribution(self, data: pd.DataFrame, title: str, axis = None):
        if axis is None:
            _, ax = plt.subplots(1,1, figsize=(7,7))
        else: 
            ax = axis

        data = data['Label'] if isinstance(data, pd.DataFrame) else data
        counts, _, patches = ax.hist(data, bins=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Background", "Signal"])
        ax.set_yticks([counts[0], counts[1]])
        patches[0].set_facecolor('royalblue') 
        patches[1].set_facecolor('mediumseagreen')
        ax.set_title(f"Event Distribution: {title}")
        if axis is None:
            plt.show()

    def plot_compared_event_distributions(self, total_data, undersampled_data, title:str):
        _, axs = plt.subplots(1, 2, figsize=(10,5))
        self.plot_event_distribution(total_data, f"{title}: Total", axs[0])
        self.plot_event_distribution(undersampled_data, f"{title}: Undersampled", axs[1])
        plt.tight_layout()
        plt.show()

    def plot_all_compared_event_distributions(self):
        self.plot_compared_event_distributions(total_data=self.total_df, undersampled_data=self.under_sampled_df, title="Dataset")
        self.plot_compared_event_distributions(total_data=self.training_df, undersampled_data=self.y_u_train, title="Training")
        self.plot_compared_event_distributions(total_data=self.valid_df, undersampled_data=self.y_u_val, title="Validation")
        self.plot_compared_event_distributions(total_data=self.test_df, undersampled_data=self.y_u_test, title="Testing")

    def undersample_data(self, ratio):
        total_df = pd.DataFrame(np.column_stack((self.X, self.y)), columns=['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id','Label'])
        self.total_df = total_df
        background_df = total_df[total_df['Label'] == 0]
        signal_df = total_df[total_df['Label'] == 1]

        undersample_count = int(len(background_df) * ratio)
        background_undersampled_df = background_df.sample(n=undersample_count, random_state=self.seed)

        under_sampled_df = pd.concat([background_undersampled_df, signal_df], axis=0)
        under_sampled_df = under_sampled_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.under_sampled_df = under_sampled_df

        X_undersampled = under_sampled_df.drop(columns='Label').to_numpy()
        y_undersampled = under_sampled_df["Label"].to_numpy()

        X_u_train, X_u_temp, y_u_train, y_u_temp = train_test_split(X_undersampled, y_undersampled, test_size=0.3, random_state=self.seed)
        X_u_val, X_u_test, y_u_val, y_u_test = train_test_split(X_u_temp, y_u_temp, test_size=0.5, random_state=self.seed)

        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=self.seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.seed)

        training_df = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id','Label'])
        valid_df = pd.DataFrame(np.column_stack((X_val, y_val)), columns=['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id','Label'])
        test_df = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id','Label'])

        self.training_df = training_df
        self.valid_df = valid_df
        self.test_df = test_df
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.X_u_train = X_u_train
        self.X_u_val = X_u_val
        self.X_u_test = X_u_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.y_u_train = y_u_train
        self.y_u_val = y_u_val
        self.y_u_test = y_u_test

        print("W/o Undersampling:")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print("---------------------------------------------------\n")
        print("W/ Undersampling:")
        print(f"Training set: {X_u_train.shape}")
        print(f"Validation set: {X_u_val.shape}")
        print(f"Test set: {X_u_test.shape}")

    @property
    def full_split_data(self):
        return self.X_train, self.X_val, self.X_test

    @property
    def full_split_labels(self):
        return self.y_train, self.y_val, self.y_test

    @property
    def undersampled_split_data(self):
        return self.X_u_train, self.X_u_val, self.X_u_test
    
    @property
    def undersampled_split_labels(self):
        return self.y_u_train, self.y_u_val, self.y_u_test





