import torch
from tqdm import tqdm


def train(device, model, train_loader, criterion, optimizer, epochs):
    model.train()
    model.to(device)
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    train_accuracy = calculate_accuracy(model, train_loader)
    print(
        "\nAccuracy on the training set after the last epoch: %.2f%%" % train_accuracy
    )


def calculate_accuracy(device, model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def validate(device, model, valid_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(valid_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        "\nAccuracy of the network on the validation images: %d %%"
        % (100 * correct / total)
    )


def test(device, model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        "\nAccuracy of the network on the test images: %d %%" % (100 * correct / total)
    )
