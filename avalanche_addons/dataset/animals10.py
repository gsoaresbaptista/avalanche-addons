import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark
from sklearn.model_selection import train_test_split


def get_animals10(
    path, n_exp, per_exp_classes=None, shuffle=True, random_state=None
):
    T = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = ImageFolder(path, transform=T)

    # stratified sampling for train and val
    train_idx, validation_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.1,
        random_state=random_state,
        shuffle=shuffle,
        stratify=dataset.targets,
    )

    # subset dataset for train and val
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, validation_idx)

    scenario = nc_benchmark(
        train_set,
        val_set,
        n_experiences=n_exp,
        shuffle=shuffle,
        seed=random_state,
        class_ids_from_zero_in_each_exp=True,
        per_exp_classes=per_exp_classes,
        task_labels=False,
    )
    return scenario
