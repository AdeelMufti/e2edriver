# evaluator.py
import os

import numpy as np
import torch
from tqdm import tqdm

import config
from metrics import calculate_ade, calculate_fde
import utils

logger = utils.get_console_logger()


class Evaluator:
    def __init__(
        self,
        model,
        data_postprocessor,
        writer,
        dry_run,
    ):
        self.model = model
        self.data_postprocessor = data_postprocessor
        self.writer = writer
        self.dry_run = dry_run

    def _save(
        self,
        data_loader_type,
        has_labels,
        batch_idx,
        names,
        gt_trajectories,
        predicted_trajectories,
        gmm_predicted_coef,
        gmm_predicted_mu,
        gmm_predicted_ln_var,
        gmm_sampled_trajectories,
        mnll_predicted_mode_probabilities,
        mnll_predicted_trajectories,
        mnll_sampled_trajectories,
    ):
        file_path = os.path.join(
            config.LOG_DIR,
            f"inferences_{data_loader_type.name}_{batch_idx}.npz",
        )
        np.savez_compressed(
            file=file_path,
            allow_pickle=True,
            names=(names),
            gt_trajectories=(gt_trajectories) if has_labels else [],
            predicted_trajectories=(predicted_trajectories),
            gmm_predicted_coef=(
                (gmm_predicted_coef) if config.GMM_TRAJECTORY_DECODER_ENABLED else []
            ),
            gmm_predicted_mu=(
                (gmm_predicted_mu) if config.GMM_TRAJECTORY_DECODER_ENABLED else []
            ),
            gmm_predicted_ln_var=(
                (gmm_predicted_ln_var) if config.GMM_TRAJECTORY_DECODER_ENABLED else []
            ),
            gmm_sampled_trajectories=(
                (gmm_sampled_trajectories)
                if config.GMM_TRAJECTORY_DECODER_ENABLED
                else []
            ),
            mnll_predicted_mode_probabilities=(
                (mnll_predicted_mode_probabilities)
                if config.MNLL_TRAJECTORY_DECODER_ENABLED
                else []
            ),
            mnll_predicted_trajectories=(
                (mnll_predicted_trajectories)
                if config.MNLL_TRAJECTORY_DECODER_ENABLED
                else []
            ),
            mnll_sampled_trajectories=(
                (mnll_sampled_trajectories)
                if config.MNLL_TRAJECTORY_DECODER_ENABLED
                else []
            ),
        )
        logger.info(
            f"Inferences saved for {data_loader_type.name} to {file_path}, length {len(names)}, batch_idx {batch_idx}."
        )

    def evaluate(self, data_loader, data_loader_type, has_labels=True, save=True):
        logger.info(f"Starting evaluation for {data_loader_type.name}")

        self.model.eval()  # Set model to evaluation mode

        total_ade = 0.0
        total_fde = 0.0
        total_ade_gmm = 0.0
        total_fde_gmm = 0.0
        total_ade_mnll = 0.0
        total_fde_mnll = 0.0

        num_batches = len(data_loader)

        loop = tqdm(
            data_loader,
            leave=True,
            desc="Evaluating",
            bar_format="{l_bar}{bar:35}{r_bar}",
        )

        names = []
        gt_trajectories = []
        predicted_trajectories = []
        gmm_predicted_coef = []
        gmm_predicted_mu = []
        gmm_predicted_ln_var = []
        gmm_sampled_trajectories = []
        mnll_predicted_mode_probabilities = []
        mnll_predicted_trajectories = []
        mnll_sampled_trajectories = []

        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, batch in enumerate(loop):
                if not batch:
                    # logger.info(f"Batch {batch_idx} is empty. Skipping...")
                    continue

                batch = self.data_postprocessor(
                    batch, training=False
                )  # Do not do noise augmentation

                if has_labels:
                    future_trajectory_gt = batch[
                        config.ProcessedDataColumn.FUTURE_STATES
                    ]
                    if save:
                        gt_trajectories.append(future_trajectory_gt.cpu().numpy())

                # Forward pass
                predicted_outputs = self.model(batch)

                # Calculate metrics
                predicted_trajectory = predicted_outputs[
                    config.PredictedDataColumn.FUTURE_STATES
                ]

                if save:
                    names.append(batch[config.ProcessedDataColumn.NAME])
                    predicted_trajectories.append(predicted_trajectory.cpu().numpy())

                if has_labels:
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

                    if save:
                        gmm_predicted_coef.append(
                            predicted_trajectory_gmm[0].cpu().numpy()
                        )
                        gmm_predicted_mu.append(
                            predicted_trajectory_gmm[1].cpu().numpy()
                        )
                        gmm_predicted_ln_var.append(
                            predicted_trajectory_gmm[2].cpu().numpy()
                        )
                        gmm_sampled_trajectories.append(
                            predicted_trajectory_gmm_sampled.cpu().numpy()
                        )

                    if has_labels:
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

                    if save:
                        mnll_predicted_mode_probabilities.append(
                            predicted_trajectory_mnll[0].cpu().numpy()
                        )
                        mnll_predicted_trajectories.append(
                            predicted_trajectory_mnll[1].cpu().numpy()
                        )
                        mnll_sampled_trajectories.append(
                            predicted_trajectory_mnll_sampled.cpu().numpy()
                        )

                    if has_labels:
                        ade_mnll = calculate_ade(
                            predicted_trajectory_mnll_sampled, future_trajectory_gt
                        )
                        fde_mnll = calculate_fde(
                            predicted_trajectory_mnll_sampled, future_trajectory_gt
                        )
                        total_ade_mnll += ade_mnll.item()
                        total_fde_mnll += fde_mnll.item()

                if has_labels:
                    log_msg_parts = []
                    log_msg_parts.append(f"ADE: {ade.item():.4f}")
                    log_msg_parts.append(f"FDE: {fde.item():.4f}")
                    loop.set_postfix_str(", ".join(log_msg_parts))

                if save and (self.dry_run or batch_idx % 100 == 0):
                    self._save(
                        data_loader_type,
                        has_labels,
                        batch_idx,
                        names,
                        gt_trajectories,
                        predicted_trajectories,
                        gmm_predicted_coef,
                        gmm_predicted_mu,
                        gmm_predicted_ln_var,
                        gmm_sampled_trajectories,
                        mnll_predicted_mode_probabilities,
                        mnll_predicted_trajectories,
                        mnll_sampled_trajectories,
                    )
                    names = []
                    gt_trajectories = []
                    predicted_trajectories = []
                    gmm_sampled_trajectories = []
                    mnll_predicted_mode_probabilities = []
                    mnll_predicted_trajectories = []
                    mnll_sampled_trajectories = []

                if self.dry_run:
                    break

        if not self.dry_run and save and len(names) > 0:
            self._save(
                data_loader_type,
                has_labels,
                batch_idx,
                names,
                gt_trajectories,
                predicted_trajectories,
                gmm_predicted_coef,
                gmm_predicted_mu,
                gmm_predicted_ln_var,
                gmm_sampled_trajectories,
                mnll_predicted_mode_probabilities,
                mnll_predicted_trajectories,
                mnll_sampled_trajectories,
            )

        if has_labels:
            avg_ade = total_ade / num_batches
            avg_fde = total_fde / num_batches
            output_str = f"Evaluation for {data_loader_type.name}. ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}"
            if config.GMM_TRAJECTORY_DECODER_ENABLED:
                avg_ade_gmm = total_ade_gmm / num_batches
                avg_fde_gmm = total_fde_gmm / num_batches
                output_str += (
                    f", GMM ADE: {avg_ade_gmm:.4f}, GMM FDE: {avg_fde_gmm:.4f}"
                )
            if config.MNLL_TRAJECTORY_DECODER_ENABLED:
                avg_ade_mnll = total_ade_mnll / num_batches
                avg_fde_mnll = total_fde_mnll / num_batches
                output_str += (
                    f", MNLL ADE: {avg_ade_mnll:.4f}, MNLL FDE: {avg_fde_mnll:.4f}"
                )
            logger.info(output_str)

            self.writer.add_scalar(f"Metrics/{data_loader_type.name}/ADE", avg_ade, 0)
            self.writer.add_scalar(f"Metrics/{data_loader_type.name}/FDE", avg_fde, 0)
            if config.GMM_TRAJECTORY_DECODER_ENABLED:
                self.writer.add_scalar(
                    f"Metrics/{data_loader_type.name}/ADE_GMM", avg_ade_gmm, 0
                )
                self.writer.add_scalar(
                    f"Metrics/{data_loader_type.name}/FDE_GMM", avg_fde_gmm, 0
                )
            if config.MNLL_TRAJECTORY_DECODER_ENABLED:
                self.writer.add_scalar(
                    f"Metrics/{data_loader_type.name}/ADE_MNLL", avg_ade_mnll, 0
                )
                self.writer.add_scalar(
                    f"Metrics/{data_loader_type.name}/FDE_MNLL", avg_fde_mnll, 0
                )

        logger.info("Evaluation completed.")
