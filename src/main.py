# main.py
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_loader import get_dataloader, DataPostprocessor
from model import EndToEndDriver
from utils import (
    setup_training_logger,
    save_checkpoint,
    load_checkpoint,
    get_model_size,
    get_console_logger,
)
from trainer import Trainer
from evaluator import Evaluator

logger = get_console_logger()


def main(args):
    # Setup logging and device
    logger.info(f"Experiment: {config.EXPERIMENT_NAME}")
    writer = setup_training_logger(config.LOG_DIR, config.CHECKPOINT_DIR)
    writer.add_text("Experiment_Name", config.EXPERIMENT_NAME, 0)
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")
    logger.info(f"Using AMP: {config.USE_AMP}")
    logger.info(f"Using Gradient Clipping: {config.GRADIENT_CLIP_MAX_NORM}")
    logger.info(f"Early State Fusion: {config.EARLY_STATE_FUSION}")

    data_postprocessor = DataPostprocessor()

    # Create DataLoaders
    logger.info("Creating data loaders...")
    train_loader = get_dataloader(
        config.TRAIN_DATASET_DIR,
        args.batch_size,
        num_data_loaders=args.num_data_loaders,
        data_loader_type=config.DataloaderType.TRAIN,
    )
    validation_loader = get_dataloader(
        config.VALIDATION_DATASET_DIR,
        args.batch_size,
        num_data_loaders=args.num_data_loaders,
        data_loader_type=config.DataloaderType.VALIDATION,
    )
    logger.info("Data loaders created.")

    # Code to sanity check data for user
    now = time.time()
    for i, batch in enumerate(train_loader):
        if not batch:
            continue

        logger.info(f"-------------------- Batch {i} --------------------")
        logger.info(f"{time.time() - now:.2f} seconds elapsed")
        logger.info("Raw batch data:")
        logger.info(
            f"NAME: {batch[config.DataColumn.NAME].dtype}, {batch[config.DataColumn.NAME].shape}"
        )
        logger.info(
            f"INTENT: {batch[config.DataColumn.INTENT].dtype}, {batch[config.DataColumn.INTENT].shape}, {batch[config.DataColumn.INTENT].device}"
        )
        logger.info(
            f"PAST_STATES_POS: {batch[config.DataColumn.PAST_STATES_POS].dtype}, {batch[config.DataColumn.PAST_STATES_POS].shape}, {batch[config.DataColumn.PAST_STATES_POS].device}"
        )
        logger.info(
            f"PAST_STATES_VEL: {batch[config.DataColumn.PAST_STATES_VEL].dtype}, {batch[config.DataColumn.PAST_STATES_VEL].shape}, {batch[config.DataColumn.PAST_STATES_VEL].device}"
        )
        logger.info(
            f"PAST_STATES_ACCEL: {batch[config.DataColumn.PAST_STATES_ACCEL].dtype}, {batch[config.DataColumn.PAST_STATES_ACCEL].shape}, {batch[config.DataColumn.PAST_STATES_ACCEL].device}"
        )
        logger.info(
            f"FUTURE_STATES_POS: {batch[config.DataColumn.FUTURE_STATES_POS].dtype}, {batch[config.DataColumn.FUTURE_STATES_POS].shape}, {batch[config.DataColumn.FUTURE_STATES_POS].device}"
        )
        for column in config.DataColumn:
            if not column.name.startswith("IMAGE"):
                continue
            logger.info(
                f"{column.name}: {batch[column].dtype}, {batch[column].shape}, {batch[column].device}"
            )

        # import sys
        # torch.set_printoptions(sci_mode=False, threshold=sys.maxsize)
        # logger.info(f"PAST_STATES_POS: {batch[config.DataColumn.PAST_STATES_POS][0]}")
        # logger.info(f"PAST_STATES_VEL: {batch[config.DataColumn.PAST_STATES_VEL][0]}")
        # logger.info(f"PAST_STATES_ACCEL: {batch[config.DataColumn.PAST_STATES_ACCEL][0]}")
        # logger.info(f"IMAGE_1_FRONT: {batch[config.DataColumn.IMAGE_1_FRONT][0][0][0]}")
        # logger.info(f"IMAGE_1_FRONT: {batch[config.DataColumn.IMAGE_1_FRONT][0][1][0]}")

        batch = data_postprocessor(batch)
        logger.info("Processed batch data:")
        logger.info(
            f"NAME: {batch[config.ProcessedDataColumn.NAME].dtype}, {batch[config.ProcessedDataColumn.NAME].shape}"
        )
        logger.info(
            f"INTENT: {batch[config.ProcessedDataColumn.INTENT].dtype}, {batch[config.ProcessedDataColumn.INTENT].shape}, {batch[config.ProcessedDataColumn.INTENT].device}"
        )
        logger.info(
            f"PAST_STATES: {batch[config.ProcessedDataColumn.PAST_STATES].dtype}, {batch[config.ProcessedDataColumn.PAST_STATES].shape}, {batch[config.ProcessedDataColumn.PAST_STATES].device}"
        )
        logger.info(
            f"FUTURE_STATES: {batch[config.ProcessedDataColumn.FUTURE_STATES].dtype}, {batch[config.ProcessedDataColumn.FUTURE_STATES].shape}, {batch[config.ProcessedDataColumn.FUTURE_STATES].device}"
        )
        for column in config.DataColumn:
            if not column.name.startswith("IMAGE"):
                continue
            image = batch[config.ProcessedDataColumn.IMAGES][column]
            logger.info(f"{column.name}: {image.dtype}, {image.shape}, {image.device}")

        break

    # Create Model
    logger.info("Creating model...")
    model = EndToEndDriver(args.dry).to(device)
    # Optional: Use DataParallel for multiple GPUs
    # if torch.cuda.device_count() > 1:
    #    logger.info(f"Using {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)
    logger.info("Model created.")

    # Code to sanity test model
    logger.info("Sanity testing model...")
    logger.info(model)
    logger.info(
        f"Model total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info(
        f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    model_size = get_model_size(model)
    logger.info(f"Model size: {model_size:,} MB")
    logger.info(f"Model device: {next(model.parameters()).device}")
    for i, batch in enumerate(train_loader):
        if not batch:
            continue
        batch = data_postprocessor(batch)
        with torch.no_grad():
            outputs = model(batch)
            logger.info(
                f"FUTURE_STATES output shape: {outputs[config.PredictedDataColumn.FUTURE_STATES].shape}"
            )
        break

    # Loss Function(s) defined in trainer.py

    # Optimizer(s)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    if config.USE_AMP:
        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler(device=config.DEVICE)
    else:
        scaler = None

    # Scheduler(s)
    # Learning Rate Scheduler (Example: ReduceLROnPlateau)
    # Adjust patience and factor as needed
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=5, factor=0.5
    # )
    # Or a step-based scheduler:
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=config.STEP_LR_STEP_SIZE, gamma=config.STEP_LR_GAMMA
    # )
    # OneCycleLR requires steps_per_epoch and epochs
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.ONE_CYCLE_MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=args.num_epochs,
    )

    if args.resume:
        start_epoch, global_step, best_metric = load_checkpoint(
            args.resume, model, optimizer, scaler, scheduler
        )
        logger.info(
            f"Resumed from epoch {start_epoch + 1}. Best metric so far: {best_metric:.4f}"
        )
    else:
        start_epoch = 0
        global_step = 0
        best_metric = float("inf")

    # Create Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        data_postprocessor=data_postprocessor,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
        writer=writer,
        checkpoint_dir=config.CHECKPOINT_DIR,
        dry_run=args.dry,
        start_epoch=start_epoch,
        global_step=global_step,
        best_metric=best_metric,
    )

    if not args.evaluate:
        # Start Training
        trainer.train(config.NUM_EPOCHS)

        # Save final model
        save_checkpoint(
            state={
                "epoch": config.NUM_EPOCHS,
                "global_step": 0,
                "best_metric": float("inf"),
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=False,
            checkpoint_dir=config.CHECKPOINT_DIR,
            step=0,
            filename="final_model.pth.tar",
        )
        logger.info("Final model saved.")

    del train_loader._iterator
    train_loader = None
    trainer.train_loader = None

    evaluator = Evaluator(
        model=model,
        data_postprocessor=data_postprocessor,
        writer=writer,
        dry_run=args.dry,
    )

    if args.evaluate and config.DataloaderType.VALIDATION.value[1]:
        evaluator.evaluate(
            data_loader=validation_loader,
            data_loader_type=config.DataloaderType.VALIDATION,
            has_labels=True,
            save=True,
        )
    del validation_loader._iterator
    validation_loader = None
    trainer.validation_loader = None

    if config.DataloaderType.TEST_HOLDOUT.value[1]:
        test_holdout_loader = get_dataloader(
            config.TRAIN_DATASET_DIR,
            args.batch_size,
            num_data_loaders=args.num_data_loaders,
            data_loader_type=config.DataloaderType.TEST_HOLDOUT,
        )
        evaluator.evaluate(
            data_loader=test_holdout_loader,
            data_loader_type=config.DataloaderType.TEST_HOLDOUT,
            has_labels=True,
            save=True,
        )
        del test_holdout_loader._iterator
        test_holdout_loader = None

    test_loader = get_dataloader(
        config.TEST_DATASET_DIR,
        args.batch_size,
        num_data_loaders=args.num_data_loaders,
        data_loader_type=config.DataloaderType.TEST,
    )
    evaluator.evaluate(
        data_loader=test_loader,
        data_loader_type=config.DataloaderType.TEST,
        has_labels=False,
        save=True,
    )
    del test_loader._iterator
    test_loader = None

    writer.close()
    logger.info("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train End-to-End Driving Model on Waymo Open Dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Input batch size for training",
    )
    parser.add_argument(
        "--num-data-loaders",
        type=int,
        default=config.NUM_DATA_LOADERS,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        default=False,
        help="If True, run a dry run without training",
    )
    parser.add_argument(
        "--shrink",
        action="store_true",
        default=False,
        help="If True, network size is shrunk for testing",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Disables CUDA training",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=config.CHECKPOINT_DIR,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=config.LOG_DIR,
        help="Directory to save logs",
    )
    parser.add_argument(
        "--train-dataset-dir",
        type=str,
        default=config.TRAIN_DATASET_DIR,
        help="Directory of the training dataset",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=config.EXPERIMENT_NAME,
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Will not train, only evaluate the model",
    )

    args = parser.parse_args()

    # Update config based on args if needed (or pass args directly)
    config.EXPERIMENT_NAME = args.experiment_name
    config.BATCH_SIZE = args.batch_size
    config.NUM_DATA_LOADERS = args.num_data_loaders
    config.LEARNING_RATE = args.learning_rate
    config.NUM_EPOCHS = args.num_epochs
    config.DEVICE = "cpu" if args.cpu else config.DEVICE
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.LOG_DIR = args.log_dir
    config.TRAIN_DATASET_DIR = args.train_dataset_dir

    if config.EXPERIMENT_NAME:
        config.CHECKPOINT_DIR = f"{config.CHECKPOINT_DIR}/{config.EXPERIMENT_NAME}"
        config.LOG_DIR = f"{config.LOG_DIR}/{config.EXPERIMENT_NAME}"

    if args.dry:
        logger.info(
            "Running in dry run mode. Only 1 batch will be trained, validated, evaled."
        )
        config.NUM_DATA_LOADERS = 1

    if args.shrink:
        config.VIT_DEPTH = 2  # Number of layers in ViT
        config.VIT_NUM_HEADS = 2  # Number of attention heads in ViT
        config.VIT_MLP_RATIO = 2.0  # MLP ratio in ViT
        config.IMAGE_EMBEDDING_DIM = 256  # Dimension of image feature vector

        config.STATE_EMBEDDING_DIM = 128  # Dimension of state feature vector
        config.STATE_LAYERS = 2  # Number of layers in state encoder

        config.TRAJECTORY_DECODER_EMBEDDING_DIM = 256
        config.TRAJECTORY_DECODER_LAYERS = 2  # Number of layers in trajectory decoder

        config.GMM_TRAJECTORY_DECODER_EMBEDDING_DIM = 256
        config.GMM_TRAJECTORY_DECODER_LAYERS = 2

        config.MNLL_TRAJECTORY_DECODER_EMBEDDING_DIM = 256
        config.MNLL_TRAJECTORY_DECODER_LAYERS = 2

    main(args)
