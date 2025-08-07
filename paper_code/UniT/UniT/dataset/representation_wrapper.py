import copy
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl



class RepresentationWrapper(pl.LightningDataModule):
    def __init__(self, dataset, cfg, train=None, validation=None, test=None):
        super().__init__()
        self.dataset_configs = dict()
        self.cfg = copy.deepcopy(cfg)
        self.dataset = dataset
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.cfg.dataloader)

    def val_dataloader(self):
        if "validation" in self.dataset_configs:
            val_dataset = self.dataset.get_validation_dataset()
            return DataLoader(val_dataset, **self.cfg.val_dataloader)
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            def __getitem__(self, idx):
                # 返回与训练数据格式匹配的空数据
                return {}  # 或根据你的数据格式调整
        return DataLoader(EmptyDataset())
    def val_dataset(self):
        if "validation" in self.dataset_configs:
            val_dataset = self.dataset.get_validation_dataset()
            return val_dataset
        return 0

    def test_dataloader(self):
        if "test" in self.dataset_configs:
            # same with validation
            val_dataset = self.dataset.get_validation_dataset()
            return DataLoader(val_dataset, **self.cfg.val_dataloader)
        return None