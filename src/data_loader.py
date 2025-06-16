# data_loader.py
import io

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import fastparquet

from transforms import (
    get_image_normalization_transforms,
    get_image_augmentation_transforms,
)
import config
import utils

logger = utils.get_console_logger()


class DataPostprocessor:
    def __init__(self):
        self.image_normalization_transforms = get_image_normalization_transforms()
        self.image_augmentation_transforms = get_image_augmentation_transforms()

    def __call__(self, batch, training=True):
        for column in config.DataColumn:
            if column.name.startswith("NAME"):
                continue
            batch[column] = batch[column].to(config.DEVICE)

        batch_processed = {}

        # Remains on CPU, not needed on GPU
        batch_processed[config.ProcessedDataColumn.NAME] = batch[config.DataColumn.NAME]

        batch_processed[config.ProcessedDataColumn.IMAGES] = {}
        for column in config.DataColumn:
            if not column.name.startswith("IMAGE"):
                continue
            image = batch[column]
            image = self.image_normalization_transforms(image)
            # Do not do noise augmentation if training is False
            image = self.image_augmentation_transforms(image) if training else image
            batch_processed[config.ProcessedDataColumn.IMAGES][column] = image

            # # Uncomment to save images for debugging
            # import os
            # from PIL import Image
            # if training:
            #     image = image.permute(0, 2, 3, 1) * 255.0
            #     image = image.to(torch.uint8)
            #     save_dir = "/tmp/saved_images"
            #     os.makedirs(save_dir, exist_ok=True)
            #     for i, img in enumerate(image):
            #         img_pil = Image.fromarray(img.cpu().numpy())
            #         img_name = f"{i}.png"
            #         img_path = os.path.join(save_dir, img_name)
            #         logger.info(f"Saving image to {img_path}")
            #         img_pil.save(img_path)

        batch_processed[config.ProcessedDataColumn.INTENT] = (
            torch.nn.functional.one_hot(
                batch[config.DataColumn.INTENT].long(),
                num_classes=config.INTENT_CLASSES,
            ).to(config.DTYPE)
        )

        batch_processed[config.ProcessedDataColumn.PAST_STATES] = torch.stack(
            [
                batch[config.DataColumn.PAST_STATES_POS],
                batch[config.DataColumn.PAST_STATES_VEL],
                batch[config.DataColumn.PAST_STATES_ACCEL],
            ],
            dim=1,
        )

        batch_processed[config.ProcessedDataColumn.FUTURE_STATES] = batch[
            config.DataColumn.FUTURE_STATES_POS
        ]

        return batch_processed


class WaymoE2EDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        batch_size=config.BATCH_SIZE,
        data_loader_type=config.DataloaderType.TRAIN,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.data_loader_type = data_loader_type
        self._init_dataset()

    def _init_dataset(self):
        logger.info(
            f"Loading dataset files from {self.dataset_dir} for {self.data_loader_type}..."
        )

        self.parquet = fastparquet.ParquetFile(self.dataset_dir)
        # create_dataset.py writes 64 records per row group, using as 1:1 for batch_size
        self.total_batches = len(self.parquet.row_groups)
        logger.info(
            f"Total batches available in data: {self.total_batches}, of batch size: {self.batch_size}, total rows: {self.parquet.count()}"
        )

        if self.data_loader_type == config.DataloaderType.TEST_HOLDOUT:
            self.start_idx = 0
            self.end_idx = int(
                self.total_batches * config.DataloaderType.TEST_HOLDOUT.value[1]
            )
        elif self.data_loader_type == config.DataloaderType.VALIDATION_HOLDOUT:
            self.start_idx = int(
                self.total_batches * config.DataloaderType.TEST_HOLDOUT.value[1]
            )
            self.end_idx = self.start_idx + int(
                self.total_batches * config.DataloaderType.VALIDATION_HOLDOUT.value[1]
            )
        elif self.data_loader_type == config.DataloaderType.TRAIN:
            self.start_idx = int(
                self.total_batches * config.DataloaderType.TEST_HOLDOUT.value[1]
            ) + int(
                self.total_batches * config.DataloaderType.VALIDATION_HOLDOUT.value[1]
            )
            self.end_idx = self.total_batches
        elif self.data_loader_type in [
            config.DataloaderType.VALIDATION,
            config.DataloaderType.TEST,
        ]:
            self.start_idx = 0
            self.end_idx = self.total_batches
        else:
            raise ValueError(
                f"Invalid data_loader_type: {self.data_loader_type}. Must be one of {config.DataloaderType.__members__.keys()}"
            )

        logger.info(
            f"Starting from batch index: {self.start_idx}, ending at: {self.end_idx}, number of batches: {len(self)}"
        )

    def __len__(self):
        return int(self.total_batches * self.data_loader_type.value[1])

    def __getitem__(self, idx):
        # logger.info(f"Loading {self.data_loader_type} dataset batch {idx}, offset to {idx + self.start_idx}...")
        idx = idx + self.start_idx

        batch = self.parquet[idx].to_pandas(columns=config.COLUMNS)
        batch_size = len(batch)

        if (
            self.data_loader_type
            in [config.DataloaderType.TRAIN, config.DataloaderType.VALIDATION_HOLDOUT]
            and batch_size <= 1
        ):
            # logger.info(f"Batch {idx} is empty. Skipping...")
            return {}

        sample = {}
        try:
            # NAME: [batch_size]
            sample[config.DataColumn.NAME] = batch[config.DataColumn.NAME.value[1]]

            # INTENT: [batch_size, 1]
            sample[config.DataColumn.INTENT] = torch.Tensor(
                batch[config.DataColumn.INTENT.value[1]]
            ).to(torch.uint8)

            # PAST_STATES_*: [batch_size, T_past=16 (4s x 4hz), 2]
            sample[config.DataColumn.PAST_STATES_POS] = torch.Tensor(
                batch[config.DataColumn.PAST_STATES_POS.value[1]]
            ).reshape(batch_size, -1, 2)
            sample[config.DataColumn.PAST_STATES_VEL] = torch.Tensor(
                batch[config.DataColumn.PAST_STATES_VEL.value[1]]
            ).reshape(batch_size, -1, 2)
            sample[config.DataColumn.PAST_STATES_ACCEL] = torch.Tensor(
                batch[config.DataColumn.PAST_STATES_ACCEL.value[1]]
            ).reshape(batch_size, -1, 2)

            # PAST_STATES_*: [batch_size, T_past=20 (5s x 4hz), 2]
            sample[config.DataColumn.FUTURE_STATES_POS] = torch.Tensor(
                batch[config.DataColumn.FUTURE_STATES_POS.value[1]]
            ).reshape(batch_size, -1, 2)

            # IMAGE_*: [batch_size, H, W, 3]
            for column in config.DataColumn:
                if not column.name.startswith("IMAGE"):
                    continue
                sample[column] = torch.from_numpy(
                    np.array(
                        [
                            np.load(io.BytesIO(batch[column.value[1]][i]))["arr_0"]
                            for i in range(batch_size)
                        ]
                    )
                )

        except Exception as e:
            logger.info(f"Error processing record {idx} in file. Skipping. Error: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed debugging

        del batch

        return sample


def get_dataloader(dataset_dir, batch_size, num_data_loaders, data_loader_type):
    """Creates a PyTorch DataLoader for the Waymo E2E dataset."""
    dataset = WaymoE2EDataset(
        dataset_dir, batch_size=batch_size, data_loader_type=data_loader_type
    )

    # Note: The initial _load_samples can be slow. Consider optimizations.
    if len(dataset) == 0:
        raise ValueError(
            f"No valid samples found in {dataset_dir}. Check dataset and parsing logic."
        )

    dataloader = DataLoader(
        dataset,
        # sampler=SequentialSampler(dataset),
        batch_size=1,
        shuffle=(
            False
            if data_loader_type != config.DataloaderType.TRAIN
            else config.SHUFFLE_TRAIN
        ),
        num_workers=num_data_loaders,
        pin_memory=True,  # Helps speed up CPU to GPU transfer
        drop_last=False,  # Drop last incomplete batch during training
        collate_fn=(lambda batch: batch[0]),
    )
    return dataloader
