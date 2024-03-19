import json

import matplotlib.pyplot as plt
import torch


def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    model.to(device)
    running_loss = 0.0
    running_accuracy = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = running_accuracy / len(train_loader.dataset)
    return epoch_loss, epoch_accuracy


def evaluate_model(device, model, criterion, test_loader):
    model.eval()
    model.to(device)
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = running_accuracy / len(test_loader.dataset)
    return test_loss, test_accuracy


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data).item()
    return corrects / labels.size(0)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def plot_loss_and_acc(df, path):
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df["Train Loss"], label="Train Loss")
    plt.plot(df["Val Loss"], label="Val Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(df["Train Accuracy"], label="Train Accuracy")
    plt.plot(df["Val Accuracy"], label="Val Accuracy")
    plt.title("Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
