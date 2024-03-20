import os
from argparse import ArgumentParser

import pandas as pd
import torch
from tqdm import tqdm

from data_preparation import get_data
from models import (
    VGG16,
    AlexNet,
    CustomCNN,
    LeNet5,
    PretrainedResNet,
    ResidualBlock,
    ResNet,
)
from train_eval_utils import evaluate_model, load_config, plot_loss_and_acc, train_epoch

MODELS = {
    "CustomCNN": CustomCNN,
    "PretrainedResNet": PretrainedResNet,
    "LeNet5": LeNet5,
    "AlexNet": AlexNet,
    "VGG16": VGG16,
    "ResNet": ResNet,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
}


def main(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cinic_directory = "."

    cinic_train, cinic_valid, cinic_test = get_data(
        cinic_directory, batch_size=256, augmentation=False
    )

    config = load_config(args.config)

    model = (
        MODELS[config["model"]["model_name"]]()
        if "ResNet" not in config["model"]["model_name"]
        else MODELS[config["model"]["model_name"]](ResidualBlock, [2, 2, 2, 2])
    )
    optimizer = OPTIMIZERS[config["training_params"]["optimizer"]](
        model.parameters(), lr=config["training_params"]["lr"]
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = SCHEDULERS[config["training_params"]["lr_scheduler"]](
        optimizer, **config["lr_scheduler_params"]
    )

    model_name = config["model"]["model_name"]
    seed = config["model"]["seed"]

    path = f"{args.checkpoints}/{model_name}_{seed}"
    os.makedirs(path, exist_ok=True)

    columns = ["Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
    df = pd.DataFrame(columns=columns)

    # Early stopping variables
    patience = config["early_stopping_params"]["patience"]
    min_delta = config["early_stopping_params"]["min_delta"]
    best_val_loss = float("inf")
    counter = 0

    num_epochs = config["training_params"]["no_epochs"]

    # Training loop with scheduler and early stopping
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_accuracy = train_epoch(
            device, model, criterion, optimizer, cinic_train
        )
        val_loss, val_accuracy = evaluate_model(device, model, criterion, cinic_valid)

        # Checkpoint saving
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            checkpoint_name = (
                f"{model_name}_epoch_{epoch}_seed_{seed}_val_loss_{val_loss:.4f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                os.path.join(path, checkpoint_name),
            )
            counter = 0
        elif val_loss >= best_val_loss - min_delta:  # Early stopping
            counter += 1
            if counter >= patience:
                print("Early stopping...")
                break

        scheduler.step()

        df.loc[epoch] = [train_loss, train_accuracy, val_loss, val_accuracy]
        print(
            f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    plot_loss_and_acc(df, f"{path}/{model_name}_{seed}.png")
    df.to_csv(f"{path}/{model_name}_{seed}.csv")

    best_model = MODELS[config["model"]["model_name"]]
    checkpoint = torch.load(f"{path}/{checkpoint_name}")
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = evaluate_model(device, best_model, criterion, cinic_test)
    with open(f"{path}/test_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--config",
        required=False,
        help="Path to the config file",
        default="config.json",
    )
    argparser.add_argument(
        "--checkpoints",
        required=False,
        help="Path to the checkpoints directory",
        default="checkpoints/",
    )
    main(argparser.parse_args())
