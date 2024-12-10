import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from helper_functions import label_dict_from_config_file, plot_loss, EarlyStopper
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import yaml
import pickle


class HandGestureDataset(Dataset):
    def __init__(self, filepath):
        self.data = torch.tensor(pd.read_csv(
            filepath).values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.list_labels = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            # First layer
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # Second layer
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Third layer
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Fourth layer
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            # Output layer
            nn.Linear(128, len(self.list_labels))
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(model, train_path, val_path, num_epochs=40, batch_size=16, lr=0.0001):
    train_loss = []
    val_loss = []
    best_model = None
    best_val_loss = float("inf")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopper(patience=20, min_delta=0.001)

    print("Loading data...")
    train_loader = DataLoader(HandGestureDataset(
        train_path), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HandGestureDataset(
        val_path), batch_size=batch_size, shuffle=True)
    print("Data loaded")
    # training loop
    for epoch in range(num_epochs):
        running_loss = 0
        for batch in train_loader:
            train_batch = batch[:, 1:]
            train_label = batch[:, 0].long()

            optimizer.zero_grad()
            # forward pass
            y_pred = model(train_batch)
            loss = criterion(y_pred, train_label)
            running_loss += loss.item()
            # backward pass
            loss.backward()
            optimizer.step()
        train_loss.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss[-1]:.4f}")

        # Validation phase
        val_loss_running = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                val_batch = batch[:, 1:]
                val_label = batch[:, 0].long()
                y_pred = model(val_batch)
                loss = criterion(y_pred, val_label)
                val_loss_running += loss.item()
        val_loss.append(val_loss_running / len(val_loader))
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            best_model = model
        print(f"Validation Loss: {val_loss[-1]:.4f}")

        # Check early stopping
        if early_stopping.early_stop(val_loss[-1]):
            print("Early stopping")
            break
        model.train()
    return best_model, train_loss, val_loss


def predict(model, test_path):
    model.eval()
    pred_labels = []
    test_labels = []
    with torch.no_grad():
        test_loader = DataLoader(HandGestureDataset(
            test_path), batch_size=1, shuffle=False)
        for batch in test_loader:
            test_batch = batch[:, 1:]
            test_label = batch[:, 0].long()
            y_pred = model(test_batch)
            pred_label = torch.argmax(y_pred, dim=1)
            pred_labels.append(pred_label.item())
            test_labels.append(test_label.item())
    print("Accuracy: ", accuracy_score(test_labels, pred_labels))
    print("Recall: ", recall_score(test_labels, pred_labels, average="macro"))
    print("Precision: ", precision_score(
        test_labels, pred_labels, average="macro"))
    print("F1: ", f1_score(test_labels, pred_labels, average="macro"))


if __name__ == "__main__":
    model = NeuralNetwork()
    model, train_loss, val_loss = train(model, "data/landmark_train.csv", "data/landmark_val.csv",
                                        num_epochs=100, batch_size=16, lr=0.0001)
    predict(model, "data/landmark_test.csv")

    print("Saving model...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Plotting loss...")
    plot_loss(train_loss, val_loss)
