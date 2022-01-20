import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from src.utils import bring_dataset_csv
from cv2 import cv2

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df["path"].values
        self.labels = df["class"].values
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image_path = self.image_id[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)
        return image, label, image_path


class ColonDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 4,
            batch_size: int = 32,
            pin_memory = False

    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Train augmentation policy
        self.train_transform = Compose(
            [
                A.RandomResizedCrop(height=self.hparams.img_size, width=self.hparams.img_size),
                A.HorizontalFlip(p=0.5),
                # Flip the input horizontally around the y-axis.
                A.ShiftScaleRotate(p=0.5),
                # Randomly apply affine transforms: translate, scale and rotate the input
                A.RandomBrightnessContrast(p=0.5),
                # Randomly change brightness and contrast of the input image.
                A.Normalize(),
                # Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                ToTensorV2(),
                # Convert image and mask to torch.Tensor
            ]
        )

        # Validation/Test augmentation policy
        self.test_transform = Compose(
            [
                A.Resize(height=self.hparams.img_size, width=self.hparams.img_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    @property
    def num_classes(self) -> int:
        return 4

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # Random train-validation split
            train_df, valid_df = bring_dataset_csv(datatype='COLON_PATCHES_1024', stage=None)

            # Train dataset
            self.train_dataset = CustomDataset(train_df, self.train_transform)
            # Validation dataset
            self.valid_dataset = CustomDataset(valid_df, self.test_transform)
            # Test dataset
        else:
            test_df = bring_dataset_csv(datatype='COLON_PATCHES_1024', stage='test')
            self.test_dataset = CustomDataset(test_df, self.test_transform)
        print("setup done!")

    def train_dataloader(self):
        print("Train Data loading")
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        print("Val data loading")
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False
        )

    def test_dataloader(self):
        print("Test data loading")
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )
