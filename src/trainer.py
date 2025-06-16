# trainer.py
import contextlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm

import config
from metrics import calculate_ade, calculate_fde
import utils
from losses import get_loss_instance

logger = utils.get_console_logger()


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        data_postprocessor,
        optimizer,
        scaler,
        scheduler,
        device,
        writer,
        checkpoint_dir,
        dry_run,
        start_epoch,
        global_step,
        best_metric,
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.data_postprocessor = data_postprocessor
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler  # Learning rate scheduler
        self.device = device
        self.writer = writer
        self.checkpoint_dir = checkpoint_dir
        self.dry_run = dry_run
        self.start_epoch = start_epoch
        self.global_step = global_step
        self.best_val_loss = best_metric

        # Instantiate loss functions based on config
        self.loss_modules = (
            nn.ModuleList()
        )  # Use ModuleList to correctly register modules
        self.loss_configurations = []
        logger.info("Initializing loss functions based on config.LOSSES:")
        for loss_conf in config.LOSSES:
            name = loss_conf[config.LossConfig.NAME]
            weight = loss_conf[config.LossConfig.WEIGHT]
            params = loss_conf.get(config.LossConfig.PARAMS, {})

            loss_function_instance = get_loss_instance(name, **params).to(self.device)
            self.loss_modules.append(loss_function_instance)
            # Store config for logging and weighting
            self.loss_configurations.append(
                {
                    config.LossConfig.NAME: loss_function_instance.name,
                    config.LossConfig.WEIGHT: weight,
                    config.LossConfig.FUNCTION: loss_function_instance,
                }
            )
            logger.info(
                f"  - Registered: {loss_function_instance.name} with weight {weight}, params: {params}"
            )

        if not self.loss_modules:
            raise ValueError(
                "LOSSES in config.py is empty or resulted in no loss modules. At least one loss function must be defined."
            )

    def _calculate_and_log_losses(
        self, predicted_outputs, ground_truth, batch, epoch, step_type
    ):
        """
        Calculates individual and total weighted loss. Logs them.
        Args:
            predicted_outputs (Tensor): Model's output.
            ground_truth (Tensor): Target values for trajectory.
            batch (dict): The full input batch (passed to loss functions if they need it).
            epoch (int): Current epoch number for logging.
            step_type (enum): config.TrainerStepType enum
        Returns:
            tuple: (total_weighted_loss, dict_of_individual_unweighted_losses)
        """
        total_weighted_loss = torch.tensor(0.0, device=self.device)
        individual_unweighted_losses = {}

        is_step = step_type in [
            config.TrainerStepType.TRAIN_STEP,
            config.TrainerStepType.VALIDATION_STEP,
        ]

        log_identifier = self.global_step if is_step else epoch

        for config_entry in self.loss_configurations:
            loss_function = config_entry[config.LossConfig.FUNCTION]
            loss_name = config_entry[config.LossConfig.NAME]
            weight = config_entry[config.LossConfig.WEIGHT]

            # Each loss function's forward can take predictions, ground_truth, and the full batch
            if loss_name == config.Loss.GMMTrajectoryLoss:
                predictions = predicted_outputs[
                    config.PredictedDataColumn.FUTURE_STATES_GMM
                ]
            elif loss_name == config.Loss.MNLLTrajectoryLoss:
                predictions = predicted_outputs[
                    config.PredictedDataColumn.FUTURE_STATES_MNLL
                ]
            else:
                predictions = predicted_outputs[
                    config.PredictedDataColumn.FUTURE_STATES
                ]
            current_unweighted_loss = loss_function(
                predictions, ground_truth, batch=batch
            )
            weighted_loss_component = weight * current_unweighted_loss

            total_weighted_loss += weighted_loss_component
            individual_unweighted_losses[loss_name] = current_unweighted_loss.item()

            if (
                step_type == config.TrainerStepType.TRAIN_STEP
                and self.global_step % config.LOG_STEP_FREQ == 0
            ):
                self.writer.add_scalar(
                    f"Loss_Components/{step_type.name}/{loss_name.name}_unweighted",
                    current_unweighted_loss.item(),
                    log_identifier,
                )
                if weight != 1.0:
                    self.writer.add_scalar(
                        f"Loss_Components/{step_type.name}/{loss_name.name}_weighted",
                        weighted_loss_component.item(),
                        log_identifier,
                    )

        if (
            step_type == config.TrainerStepType.TRAIN_STEP
            and self.global_step % config.LOG_STEP_FREQ == 0
        ):
            self.writer.add_scalar(
                f"Loss_Total/{step_type.name}/Total_Weighted_Loss",
                total_weighted_loss.item(),
                log_identifier,
            )

        return total_weighted_loss, individual_unweighted_losses

    def train_epoch(self, epoch):
        self.model.train()  # Set model to training mode

        loop = tqdm(self.train_loader, leave=True, bar_format="{l_bar}{bar:35}{r_bar}")

        epoch_total_weighted_loss_sum = 0.0
        epoch_individual_unweighted_losses_sum = {
            loss_config[config.LossConfig.NAME]: 0.0
            for loss_config in self.loss_configurations
        }

        num_batches = len(self.train_loader)

        # # Code to accumulate gradients for computing gradient norm
        # grad_norms = []

        for batch_idx, batch in enumerate(loop):
            if not batch:
                logger.info(f"Batch {self.global_step} is empty. Skipping...")
                continue

            batch = self.data_postprocessor(batch)

            # Ground truth
            future_trajectory_gt = batch[config.ProcessedDataColumn.FUTURE_STATES]

            # Zero gradients
            self.optimizer.zero_grad()

            with (
                autocast(device_type=config.DEVICE)
                if config.USE_AMP
                else contextlib.nullcontext()
            ):
                # Forward pass
                predicted_outputs = self.model(batch)

                # Calculate loss
                loss, individual_losses_items = self._calculate_and_log_losses(
                    predicted_outputs,
                    future_trajectory_gt,
                    batch,
                    epoch,
                    config.TrainerStepType.TRAIN_STEP,
                )

            if torch.isnan(loss).any():
                logger.info("Error: NaN loss.")
                raise SystemExit("NaN Loss")

            # # Code to accumulate gradients for computing gradient norm
            # # Collect gradients
            # grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            # # Concatenate gradients
            # all_grads = torch.cat(grads)
            # # Compute norm
            # grad_norm = torch.linalg.norm(all_grads).item()
            # grad_norms.append(grad_norm)

            # Gradient Clipping
            # Computed over 1000 steps with StepLR: 118406.4688
            if config.GRADIENT_CLIP_MAX_NORM:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=config.GRADIENT_CLIP_MAX_NORM
                )

            # Optimizer step
            skip_lr_sched = False
            if config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                scale = self.scaler.get_scale()
                self.scaler.update()
                skip_lr_sched = scale != self.scaler.get_scale()
            else:
                loss.backward()
                self.optimizer.step()

            if not skip_lr_sched and isinstance(
                self.scheduler, optim.lr_scheduler.OneCycleLR
            ):
                # OneCycleLR requires a step after each optimizer step
                self.scheduler.step()

            self.global_step += 1

            epoch_total_weighted_loss_sum += loss.item()
            for name, val in individual_losses_items.items():
                epoch_individual_unweighted_losses_sum[name] += val

            # Log training loss periodically
            if self.global_step % config.LOG_STEP_FREQ == 0:
                log_msg_parts = [f"TotalLoss: {loss.item():.4f}"]
                # for name, val in individual_losses_items.items():
                #     log_msg_parts.append(f"{name}: {val:.4f}")
                loop.set_postfix_str(
                    f"{', '.join(log_msg_parts)}; LR: {self.optimizer.param_groups[0]['lr']:.1e}"
                )
                self.writer.add_scalar(
                    "LearningRate/step",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step,
                )

            if self.dry_run:
                break

        # # Code to accumulate gradients for computing gradient norm
        # avg_grad_norm = torch.sum(grad_norms) / len(grad_norms)
        # logger.info(f"Avg Grad Norm: {avg_grad_norm:.4f}")

        avg_epoch_total_weighted_loss = epoch_total_weighted_loss_sum / num_batches
        logger.info(
            f"Epoch {epoch} Train Avg Total Weighted Loss: {avg_epoch_total_weighted_loss:.4f}"
        )
        self.writer.add_scalar(
            f"Loss_Total/{config.TrainerStepType.TRAIN_EPOCH.name}/Total_Weighted_Loss",
            avg_epoch_total_weighted_loss,
            epoch,
        )
        for name, total_val in epoch_individual_unweighted_losses_sum.items():
            avg_val = total_val / num_batches
            self.writer.add_scalar(
                f"Loss_Components/{config.TrainerStepType.TRAIN_EPOCH.name}/{name.name}_unweighted",
                avg_val,
                epoch,
            )
            logger.info(f"  Avg Train {name.name} (unweighted): {avg_val:.4f}")
        self.writer.add_scalar(
            "LearningRate/epoch", self.optimizer.param_groups[0]["lr"], epoch
        )

        # Step the scheduler (if based on epochs)
        if self.scheduler:
            # Check if scheduler steps per epoch or per iteration
            # Example for ReduceLROnPlateau which needs validation metric
            # Or StepLR which steps per epoch
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                pass  # Step is called in validate()
            elif isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                self.scheduler.step()
            else:
                pass

    def validate(self, epoch):
        self.model.eval()  # Set model to evaluation mode

        epoch_total_weighted_loss_sum = 0.0
        epoch_individual_unweighted_losses_sum = {
            cfg[config.LossConfig.NAME]: 0.0 for cfg in self.loss_configurations
        }

        total_ade = 0.0
        total_fde = 0.0
        total_ade_gmm = 0.0
        total_fde_gmm = 0.0
        total_ade_mnll = 0.0
        total_fde_mnll = 0.0

        num_batches = len(self.validation_loader)

        loop = tqdm(
            self.validation_loader,
            leave=True,
            desc="Validating",
            bar_format="{l_bar}{bar:35}{r_bar}",
        )

        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, batch in enumerate(loop):
                if not batch:
                    logger.info(f"Batch {self.global_step} is empty. Skipping...")
                    continue

                batch = self.data_postprocessor(
                    batch, training=False
                )  # Do not add noise

                future_trajectory_gt = batch[config.ProcessedDataColumn.FUTURE_STATES]

                # Forward pass
                predicted_outputs = self.model(batch)

                # Calculate loss
                loss, individual_losses_items = self._calculate_and_log_losses(
                    predicted_outputs,
                    future_trajectory_gt,
                    batch,
                    epoch,
                    config.TrainerStepType.VALIDATION_STEP,
                )

                epoch_total_weighted_loss_sum += loss.item()
                for name, val in individual_losses_items.items():
                    epoch_individual_unweighted_losses_sum[name] += val

                # Calculate metrics
                predicted_trajectory = predicted_outputs[
                    config.PredictedDataColumn.FUTURE_STATES
                ]
                ade = calculate_ade(predicted_trajectory, future_trajectory_gt)
                fde = calculate_fde(predicted_trajectory, future_trajectory_gt)
                total_ade += ade.item()
                total_fde += fde.item()

                if config.GMM_TRAJECTORY_DECODER_ENABLED:
                    predicted_trajectory_gmm = predicted_outputs[
                        config.PredictedDataColumn.FUTURE_STATES_GMM
                    ]
                    predicted_trajectory_gmm_sampled = utils.sample_trajectory_from_gmm(
                        *predicted_trajectory_gmm
                    )
                    ade_gmm = calculate_ade(
                        predicted_trajectory_gmm_sampled, future_trajectory_gt
                    )
                    fde_gmm = calculate_fde(
                        predicted_trajectory_gmm_sampled, future_trajectory_gt
                    )
                    total_ade_gmm += ade_gmm.item()
                    total_fde_gmm += fde_gmm.item()

                if config.MNLL_TRAJECTORY_DECODER_ENABLED:
                    predicted_trajectory_mnll = predicted_outputs[
                        config.PredictedDataColumn.FUTURE_STATES_MNLL
                    ]
                    predicted_trajectory_mnll_sampled = (
                        utils.sample_trajectory_from_mnll(*predicted_trajectory_mnll)
                    )
                    ade_mnll = calculate_ade(
                        predicted_trajectory_mnll_sampled, future_trajectory_gt
                    )
                    fde_mnll = calculate_fde(
                        predicted_trajectory_mnll_sampled, future_trajectory_gt
                    )
                    total_ade_mnll += ade_mnll.item()
                    total_fde_mnll += fde_mnll.item()

                log_msg_parts = [f"TotalLoss: {loss.item():.4f}"]
                # for name, val in individual_losses_items.items(): # Can be verbose for val step
                #     log_msg_parts.append(f"{name}: {val:.4f}")
                log_msg_parts.append(f"ADE: {ade.item():.4f}")
                log_msg_parts.append(f"FDE: {fde.item():.4f}")
                loop.set_postfix_str(", ".join(log_msg_parts))

                if self.dry_run:
                    break

        avg_epoch_total_weighted_loss = epoch_total_weighted_loss_sum / len(
            self.validation_loader
        )
        avg_ade = total_ade / num_batches
        avg_fde = total_fde / num_batches
        output_str = f"Epoch {epoch} Val Avg Total Weighted Loss: {avg_epoch_total_weighted_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}"
        if config.GMM_TRAJECTORY_DECODER_ENABLED:
            avg_ade_gmm = total_ade_gmm / num_batches
            avg_fde_gmm = total_fde_gmm / num_batches
            output_str += f", GMM ADE: {avg_ade_gmm:.4f}, GMM FDE: {avg_fde_gmm:.4f}"
        if config.MNLL_TRAJECTORY_DECODER_ENABLED:
            avg_ade_mnll = total_ade_mnll / num_batches
            avg_fde_mnll = total_fde_mnll / num_batches
            output_str += (
                f", MNLL ADE: {avg_ade_mnll:.4f}, MNLL FDE: {avg_fde_mnll:.4f}"
            )
        logger.info(output_str)

        self.writer.add_scalar(
            f"Loss_Total/{config.TrainerStepType.VALIDATION_EPOCH.name}/Total_Weighted_Loss",
            avg_epoch_total_weighted_loss,
            epoch,
        )
        for name, total_val in epoch_individual_unweighted_losses_sum.items():
            avg_val = total_val / len(self.validation_loader)
            self.writer.add_scalar(
                f"Loss_Components/{config.TrainerStepType.VALIDATION_EPOCH.name}/{name.name}_unweighted",
                avg_val,
                epoch,
            )
            logger.info(f"  Avg Val {name.name} (unweighted): {avg_val:.4f}")
        self.writer.add_scalar("Metrics/Validation/ADE", avg_ade, epoch)
        self.writer.add_scalar("Metrics/Validation/FDE", avg_fde, epoch)
        if config.GMM_TRAJECTORY_DECODER_ENABLED:
            self.writer.add_scalar("Metrics/Validation/ADE_GMM", avg_ade_gmm, epoch)
            self.writer.add_scalar("Metrics/Validation/FDE_GMM", avg_fde_gmm, epoch)
        if config.MNLL_TRAJECTORY_DECODER_ENABLED:
            self.writer.add_scalar("Metrics/Validation/ADE_MNLL", avg_ade_mnll, epoch)
            self.writer.add_scalar("Metrics/Validation/FDE_MNLL", avg_fde_mnll, epoch)

        # Step scheduler like ReduceLROnPlateau
        if self.scheduler and isinstance(
            self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(avg_epoch_total_weighted_loss)

        # Save checkpoint logic
        is_best = avg_epoch_total_weighted_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_epoch_total_weighted_loss
            logger.info(f"New best model found with val loss: {self.best_val_loss:.4f}")

        if epoch % config.SAVE_CHECKPOINT_FREQ == 0 or is_best:
            utils.save_checkpoint(
                state={
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "best_metric": self.best_val_loss,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                step=self.global_step,
            )

        return avg_epoch_total_weighted_loss

    def train(self, num_epochs):

        # Log initial learning rate
        self.writer.add_scalar(
            "LearningRate/epoch", self.optimizer.param_groups[0]["lr"], 0
        )

        logger.info(f"Starting training from epoch {self.start_epoch}...")
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            logger.info(f"\n--- Epoch {epoch}/{num_epochs} ---")

            self.train_epoch(epoch)

            current_val_loss = self.validate(epoch)  # Validate after each epoch

            logger.info(
                f"Epoch {epoch} Validation Loss: {current_val_loss:.4f}, Best Loss: {self.best_val_loss:.4f}"
            )

            if self.dry_run:
                break

        logger.info("Training finished.")
