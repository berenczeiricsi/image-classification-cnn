import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from architecture import CNN
from utils import set_seeds, select_device
from dataset import ImagesDataset


def main(
        results_path
):
    set_seeds(42)

    device = select_device()

    os.makedirs(results_path, exist_ok=True)

    transform_chain = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    dataset = ImagesDataset(image_dir='training_data')
    augmented_dataset = ImagesDataset(image_dir='training_data', augmentations=transform_chain)

    rng = np.random.default_rng(seed=42)
    n_samples = len(dataset)
    shuffled_indices = rng.permutation(n_samples)

    num_validation = int(n_samples / 5)
    num_test = int(n_samples / 5)
    training_indices, validation_indices, test_indices = np.array_split(shuffled_indices,[n_samples - num_validation - num_test, n_samples - num_test])

    training_set = Subset(dataset, training_indices.tolist())
    augmented_training_set = Subset(augmented_dataset, training_indices.tolist())
    validation_set = Subset(dataset, validation_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())

    training_loader = DataLoader(training_set, shuffle=True, batch_size=32)
    augmented_training_loader = DataLoader(augmented_training_set, shuffle=True, batch_size=32)
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=32)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=32)

