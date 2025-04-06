import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader # type: ignore

from sklearn.metrics import accuracy_score, confusion_matrix # type: ignore
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

class ResultsBuilder:
    def __init__(self, 
                 test_dataloader: DataLoader, 
                 test_u_dataloader: DataLoader, 
                 device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.test_dataloader = test_dataloader
        self.test_u_dataloader = test_u_dataloader
        self.device = device
        self.accuracies = {}

    def calculate_accuracy(self, model: nn.Module, undersampled: bool = False):
        model.eval()
        model.to(self.device)
        all_predictions = []
        all_labels = []
        description = f"Accuracy Calculation: {model.__class__.__name__}"

        tester = self.test_dataloader if undersampled == False else self.test_u_dataloader

        with torch.no_grad():
            for events, labels in tqdm(tester, desc=description):
                outputs = model(events)
                _, predicted = torch.max(outputs, 1)

                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        accuracy = accuracy_score(all_labels, all_predictions)
        tst_str = "Full Dataset Test Accuracy:" if undersampled == False else "Undersampled Dataset Test Accuracy:"
        print(f"{tst_str} {accuracy * 100:.2f}%")

        cm = confusion_matrix(all_predictions, all_labels)
        return accuracy, cm
    
    def plot_losses(self, training_losses, validation_losses, title:str, axis = None):
        if axis is None:
            _, ax = plt.subplots(figsize=(7,7))
        else:
            ax = axis
        x = range(1, len(training_losses) + 1)
        ax.plot(x, training_losses, label='Training loss')
        ax.plot(x, validation_losses, label='Validation loss')
        ax.legend()
        ax.set_title("Loss over epochs")
        if axis is None:
            plt.show()

    def plot_confusion_matrix(self, cm, title:str, axis=None):
        if axis is None:
            _, ax = plt.subplots(figsize=(7, 7))
        else:
            ax = axis
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Background", "Signal"], yticklabels=["Background", "Signal"], ax=ax)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(title)

        if axis is None:
            plt.show()

    def plot_model_performance(self, model: nn.Module, model_name: str, train_loss, val_loss):
        accuracy, cm = self.calculate_accuracy(model)
        _, u_cm = self.calculate_accuracy(model, undersampled=True)

        fig, axs = plt.subplots(1,3, figsize=(30,10))
        self.plot_confusion_matrix(cm, title="Confusion Matrix: Full Dataset", axis=axs[1])
        self.plot_losses(train_loss, val_loss, title="Loss over epochs", axis=axs[0])
        self.plot_confusion_matrix(u_cm, title="Confusion Matrix: Undersampled Dataset", axis=axs[2])

        fig.suptitle(f"{model_name} Architecture", fontsize=24)
        plt.tight_layout()
        plt.show()

        self.accuracies[f"{model_name}"] = accuracy

    def plot_accuracy_by_architecture(self):
        keys = list(self.accuracies.keys())
        values = list(self.accuracies.values())

        plt.bar(keys, values)

        plt.xlabel('Architecture')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Model Architecture')

        plt.show()