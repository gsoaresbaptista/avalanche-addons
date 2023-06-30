import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark


def get_animals10():
    T = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = ImageFolder(
        "/home/gabriel/Documents/github/data/archive/raw-img", transform=T
    )
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
    )
    scenario = nc_benchmark(
        train_set,
        val_set,
        n_experiences=5,
        shuffle=True,
        seed=1234,
        class_ids_from_zero_in_each_exp=True,
        task_labels=False,
    )
    return scenario
