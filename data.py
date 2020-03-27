import numpy as np
from torchvision.datasets import CIFAR10

class SubCIFAR(CIFAR10):
    def __init__(self, *args, filter_classes=[], **kwargs):
        super().__init__(*args, **kwargs)

        data, labels = self._filter_by_labels(self.data, self.targets, filter_classes)
        self.data = data
        self.targets = labels

    def _filter_by_labels(self, data, labels, filter_classes):
        labels = np.array(labels)
        mask = np.in1d(labels, filter_classes)
        return data.compress(mask, axis=0), labels[mask].tolist()

class NoLabels:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]