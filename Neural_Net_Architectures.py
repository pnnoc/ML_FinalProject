import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader

class ShallowNarrowModel(nn.Module):
    def __init__(self):
        super(ShallowNarrowModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(11,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2),
        )

    def forward(self, x):
        return self.layers(x)
    
class ShallowWideModel(nn.Module):
    def __init__(self):
        super(ShallowWideModel, self).__init__()
        
        self.dropout_probability = 0.5
        self.layers = nn.Sequential(
            nn.Linear(11,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2),
        )

    def forward(self, x):
        return self.layers(x)
    
class DeepNarrowModel(nn.Module):
    def __init__(self):
        super(DeepNarrowModel, self).__init__()
        self.dropout_probability = 0.5

        self.layers = nn.Sequential(
            nn.Linear(11,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2),
        )

    def forward(self, x):
        return self.layers(x)
    
class DeepWideModel(nn.Module):
    def __init__(self):
        super(DeepWideModel, self).__init__()
        self.dropout_prob = 0.5
        self.layers = nn.Sequential(
            nn.Linear(11,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2),
        )

    def forward(self, x):
        return self.layers(x)
    
class Trainer: 
    def __init__(self, 
                 training_dataloader: DataLoader, 
                 training_u_dataloader: DataLoader, 
                 validation_dataloader: DataLoader, 
                 validation_u_dataloader: DataLoader, 
                 device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        self.training_dataloader = training_dataloader
        self.training_u_dataloader = training_u_dataloader
        self.validation_dataloader = validation_dataloader
        self.validation_u_dataloader = validation_u_dataloader
        self.device = device

    def train_model(self, model: nn.Module, optimizer, num_epochs = 5, criterion = nn.CrossEntropyLoss(), undersampled:bool = False):
        training_losses, validation_losses = [], []
        model.to(self.device)

        trainer_loader = self.training_dataloader if undersampled == False else self.training_u_dataloader
        validator_loader = self.validation_dataloader if undersampled == False else self.validation_u_dataloader

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for events, labels in tqdm(trainer_loader, desc=f"Training Loop: {model.__class__.__name__}"):
                events, labels = events.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(events)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)

            train_loss = running_loss / len(trainer_loader.dataset)
            training_losses.append(train_loss)

            model.eval()
            running_loss = 0.0

            with torch.no_grad():
                for events, labels in tqdm(validator_loader, desc=f"Validation Loop: {model.__class__.__name__}"):
                    events, labels = events.to(self.device), labels.to(self.device)
                    outputs = model(events)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)

                val_loss = running_loss / len(validator_loader.dataset)
                validation_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}\n\n")
        
        return training_losses, validation_losses