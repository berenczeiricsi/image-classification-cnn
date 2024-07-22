import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from architecture import CNN
from utils import set_seeds, select_device, evaluate_model, plot
from dataset import ImagesDataset


def main(
        results_path,
        network_config: dict,
        learning_rate: int,
        weight_decay: int,
        num_epochs: int
):
    """
    Main function for training and evaluation the Convolutional Neural Network
    :param results_path: Path to save the results and the trained model
    :param network_config: Dictionary containing network configuration parameters
    :param learning_rate: Learning rate for the optimizer
    :param weight_decay: Weight decay for the optimizer
    :param num_epochs: Number of epochs to train the model
    :return: None
    """
    # Set the seed for reproducibility
    set_seeds(42)

    # Select target device
    device = select_device()

    # Create a directory for the results
    os.makedirs(results_path, exist_ok=True)

    # Define augmentations for training
    transform_chain = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    # Load the dataset with and without augmentations
    dataset = ImagesDataset(image_dir='training_data')
    augmented_dataset = ImagesDataset(image_dir='training_data', augmentations=transform_chain)

    # Get random indices for the separation of the dataset
    rng = np.random.default_rng(seed=42)
    n_samples = len(dataset)
    shuffled_indices = rng.permutation(n_samples)
    num_validation = int(n_samples / 5)
    num_test = int(n_samples / 5)
    training_indices, validation_indices, test_indices = np.array_split(shuffled_indices,
                                                                        [n_samples - num_validation - num_test,
                                                                         n_samples - num_test])

    # Create the training (with and without augmentation), evaluation and test set
    training_set = Subset(dataset, training_indices.tolist())
    augmented_training_set = Subset(augmented_dataset, training_indices.tolist())
    validation_set = Subset(dataset, validation_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())

    # Create data loaders for all the sets
    training_loader = DataLoader(training_set, shuffle=True, batch_size=32)
    augmented_training_loader = DataLoader(augmented_training_set, shuffle=True, batch_size=32)
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=32)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=32)

    # Define the CNN architecture
    model = CNN(**network_config).to(device)

    # Select loss function
    criterion = nn.CrossEntropyLoss()

    # Select optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Create and save the model with the extension '.pth'
    trained_model = os.path.join(results_path, "model.pth")
    torch.save(model.state_dict(), trained_model)

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for data in tqdm(augmented_training_loader, desc="Training", position=0, leave=False):
            inputs, targets, *_ = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(augmented_training_loader)
        train_losses.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(validation_loader, desc="Evaluating", position=0, leave=False):
                inputs, targets, *_ = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        epoch_val_loss = running_val_loss / len(validation_loader)
        val_losses.append(epoch_val_loss)

        val_accuracy = 100 * correct / total
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        scheduler.step(epoch_val_loss)

        # Save model with the lowest validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), trained_model)
        else:
            epochs_no_improve += 1

        # Apply early stopping, if accuracy doesn't improve
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

    print('Training finished')

    # Plot the training and evaluation losses
    plot(train_losses, val_losses, results_path)
    print(f"You can find a plot of the training and evaluation losses in the directory: {results_path}")

    # Load the best model and evaluate it on the train, evaluation and test set
    print(f"Computing scores for best model")
    model.load_state_dict(torch.load(trained_model))

    train_loss = evaluate_model(model, training_loader, criterion, device, 'training set')
    val_loss = evaluate_model(model, validation_loader, criterion, device, 'validation set')
    test_loss = evaluate_model(model, test_loader, criterion, device, 'test set')

    print(f"Scores:")
    print(f"training loss:   {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"test loss:       {test_loss}")

    # Save scores to a txt file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"training loss:   {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"test loss:       {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="working_config.json")
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
