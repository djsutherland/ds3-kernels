from torch.utils import data
from torchvision.datasets import Omniglot
from torchvision.transforms import RandomRotation


class CombinedOmniglot(data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        rotations=(0, 90, 180, 270),
        download=False,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.rotations = rotations
        self.n_rotations = len(rotations)

        self.n_per_class = 20
        self._bg = Omniglot(root=root, background=True, download=download)
        self._bg_n_classes = n_bg = len(self._bg._character_images)
        self._eval = Omniglot(
            root=root,
            background=False,
            target_transform=lambda t: t + n_bg,
            download=download,
        )
        self.n_base_classes = n_bg + len(self._eval._character_images)
        self.base = data.ConcatDataset([self._bg, self._eval])

    def decompose_class_id(self, class_i):
        return class_i // self.n_rotations, class_i % self.n_rotations

    def construct_class_id(self, base_num, rotation_num):
        assert 0 <= base_num <= self.n_base_classes
        assert 0 <= rotation_num <= self.n_rotations
        return base_num * self.n_rotations + rotation_num

    def __getitem__(self, key):
        rotation_i = key % self.n_rotations
        base_i = key // self.n_rotations
        img, y = self.base[base_i]

        rotation = self.rotations[rotation_i]
        img = RandomRotation([rotation, rotation])(img)
        if self.transform:
            img = self.transform(img)

        y = y * self.n_rotations + rotation_i
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def __len__(self):
        return self.n_rotations * len(self.base)

    def class_subset(self, i):
        base_i, rotation_i = self.decompose_class_id(i)
        start = self.n_per_class * self.n_rotations * base_i + rotation_i
        end = start + self.n_per_class * self.n_rotations
        return data.Subset(self, range(start, end, self.n_rotations))
