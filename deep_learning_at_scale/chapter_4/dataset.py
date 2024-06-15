import os
from itertools import chain

import numpy as np
import torch
import torchvision
from datasets import load_dataset
from lightning import LightningDataModule
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, default_data_collator

ImageFile.LOAD_TRUNCATED_IMAGES = True

validation_split_percentage: int = 5


class WikiDataModule(LightningDataModule):
    def __init__(
        self,
        name: str = "gpt2",
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset_name = "wikitext"
        self.dataset_config_name = "wikitext-2-raw-v1"
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.config = AutoConfig.from_pretrained(name)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            use_fast=True,
        )

    def setup(self, stage: str):
        column_names = list(self.raw_datasets["train"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            output = self.tokenizer(examples[text_column_name])
            return output

        tokenized_datasets = self.raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=None if self.num_workers == 0 else self.num_workers,
        )
        block_size = self.tokenizer.model_max_length

        # Main data processing function that will concatenate all texts from our
        # dataset and generate chunks of block_size. It concatenated texts and
        # drop the remainder that dont fit in multiple of block size. Padding
        # can be used to avoid drop but for simplicity, no padding is used.
        def group_texts(examples):
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        self.lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=None if self.num_workers == 0 else self.num_workers,
        )

    def prepare_data(self):
        self.raw_datasets = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
        )
        self.raw_datasets["validation"] = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
        )
        self.raw_datasets["train"] = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
        )

    def train_dataloader(self):
        return DataLoader(
            self.lm_datasets["train"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.lm_datasets["validation"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.lm_datasets["test"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class SceneParsingDataset(Dataset):
    def __init__(self, set: str = "train", transform=None, target_transform=None):
        # https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
        # has  150 classes for segmentations, plus one for background given by id 0.
        self.dataset_name = "scene_parse_150"
        self.train = True
        self.transform = transform
        self.target_transform = target_transform
        raw_datasets = load_dataset(self.dataset_name)
        self.dataset = raw_datasets[set]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[
            int(idx)
        ]  # The int wrapping here is added to support FFCV use as done in chapter 7.
        image, label = record["image"], record["annotation"]

        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class PILToTensorUnScaled:
    def __init__(self, num_classes: int = 150) -> None:
        self.num_classes = num_classes

    def __call__(
        self,
        image,
    ):
        image = torch.as_tensor(np.array(image), dtype=torch.int32)
        # If 255 is used to represent unknown/other, make them background
        image[image > self.num_classes] = 0
        return image


class SceneParsingModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 85,
        num_workers: int = 4,
        input_size: int = 256,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                # ),
                torchvision.transforms.Resize((input_size, input_size), antialias=True),
            ]
        )
        self.target_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(input_size, input_size),
                    interpolation=0,  # InterpolationMode.NEAREST,
                    antialias=True,
                ),
                PILToTensorUnScaled(),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train = SceneParsingDataset(
                set="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.validation = SceneParsingDataset(
                set="validation",
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            self.test = SceneParsingDataset(
                set="test",
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def prepare_data(self): ...

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
        )
