# losses.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MSETrajectoryLoss(nn.Module):
    def __init__(self, name=config.Loss.MSETrajectoryLoss):
        super().__init__()
        self.name = name

    def forward(self, predictions, ground_truth, batch=None):
        """
        Calculates Mean Squared Error for trajectory points.
        Args:
            predictions (Tensor): Predicted trajectories (B, Seq_len, Point_dim).
            ground_truth (Tensor): Ground truth trajectories (B, Seq_len, Point_dim).
        Returns:
            Tensor: Scalar MSE loss value.
        """
        return F.mse_loss(predictions, ground_truth)


class L1TrajectoryLoss(nn.Module):
    def __init__(self, name=config.Loss.L1TrajectoryLoss):
        super().__init__()
        self.name = name

    def forward(self, predictions, ground_truth, batch=None):
        """
        Calculates L1 (Mean Absolute Error) for trajectory points.
        Args:
            predictions (Tensor): Predicted trajectories (B, Seq_len, Point_dim).
            ground_truth (Tensor): Ground truth trajectories (B, Seq_len, Point_dim).
        Returns:
            Tensor: Scalar L1 loss value.
        """
        return F.l1_loss(predictions, ground_truth)


class GMMTrajectoryLoss(nn.Module):
    def __init__(
        self, name=config.Loss.GMMTrajectoryLoss, num_components=config.GMM_COMPONENTS
    ):
        super().__init__()
        self.name = name
        self.num_components = num_components

    def forward(self, predictions, ground_truth, batch=None):
        eps = 1e-10

        # coef: torch.Size([64, 4]), mu: torch.Size([64, 4, 20, 2]), ln_var: torch.Size([64, 4, 20, 2])
        coef, mu, ln_var = predictions

        # Formulation: Adeel's 2019 Blog, Half Gaussian
        # http://blog.adeel.io/2019/06/18/context-aware-prediction-with-uncertainty-of-traffic-behaviors-using-mixture-density-networks/
        # x_gt: torch.Size([64, 1, 20]), y_gt: torch.Size([64, 1, 20]), x_pred: torch.Size([64, 4, 20]), y_pred: torch.Size([64, 4, 20]
        x_gt = ground_truth[..., 0].unsqueeze(1)
        y_gt = ground_truth[..., 1].unsqueeze(1)
        x_pred = mu[..., 0]
        y_pred = mu[..., 1]
        # displacement_sq: torch.Size([64, 4, 20, 1]), var: torch.Size([64, 4, 20, 2])
        displacement_sq = ((x_gt - x_pred) ** 2) + ((y_gt - y_pred) ** 2)
        displacement_sq = displacement_sq.unsqueeze(-1)
        min_val = (
            int((torch.log(torch.tensor(torch.finfo(config.DTYPE).tiny))).item()) / 2.0
        )
        ln_var = torch.clip(
            ln_var, min_val * 1.0, min_val * -1.0
        )  # Numerical stability
        var = torch.exp(ln_var)
        coef = F.softmax(coef, dim=1).unsqueeze(-1)  # torch.Size([64, 4, 1])
        part_a = displacement_sq / (2 * var)
        part_b = torch.log(torch.sqrt(2 * math.pi * var))
        gaussian = torch.sum(part_a + part_b, dim=2)
        gmm_loss = torch.sum(coef * gaussian)

        ### Entropy Regularization Loss ###
        # To encourage diversity in the coefficients
        entropy_per_mode = -coef * torch.log(coef + eps)
        entropy_per_batch_item = torch.sum(entropy_per_mode, dim=1)
        entropy_loss = torch.mean(-entropy_per_batch_item)

        loss = gmm_loss + entropy_loss

        return loss


class MNLLTrajectoryLoss(nn.Module):
    def __init__(
        self, name=config.Loss.MNLLTrajectoryLoss, num_modes=config.MNLL_MODES
    ):
        super().__init__()
        self.name = name
        self.num_modes = num_modes

    def forward(self, predictions, ground_truth, batch=None):
        eps = 1e-10

        mode_probability, predicted_trajectory = predictions
        mode_probability = F.softmax(mode_probability, dim=1)

        B, M, T, _ = predicted_trajectory.shape

        ### Entropy Regularization Loss ###
        # To encourage diversity in the predicted modes
        entropy_per_mode = -mode_probability * torch.log(mode_probability + eps)
        entropy_per_batch_item = torch.sum(entropy_per_mode, dim=1)
        entropy_loss = torch.mean(-entropy_per_batch_item)

        ### Min NLL Loss ###
        # Expand ground_truth for broadcasting to compare with all modes
        # Shape: (B, 1, T, 2)
        ground_truth_expanded = ground_truth.unsqueeze(1)

        # 1. Calculate L2 distance (or squared L2) for finding the closest mode
        # Shape: (B, M)
        squared_diff = (predicted_trajectory - ground_truth_expanded) ** 2
        l2_distances_sq = torch.sum(squared_diff, dim=[-1, -2])  # Sum over T and 2

        # Find the index of the mode closest to the ground truth for each batch item
        # Shape: (B,)
        min_l2_indices = torch.argmin(l2_distances_sq, dim=1)

        # 2. Calculate the NLL for the probability of the closest mode
        # Gather the predicted probabilities for the closest mode for each batch item
        # Shape: (B,)
        predicted_prob_of_closest_mode = torch.gather(
            mode_probability, 1, min_l2_indices.unsqueeze(1)
        ).squeeze(1)

        # Add a small epsilon to prevent log(0)
        nll_prob_loss_per_batch_item = -torch.log(predicted_prob_of_closest_mode + eps)

        # 3. Calculate the Regression Loss for the best matching mode
        # Select the predicted trajectory that corresponds to the closest mode
        # Use advanced indexing: predicted_trajectory[batch_indices, mode_indices]
        # Shape: (B, T, 2)
        best_pred_trajectory = predicted_trajectory[torch.arange(B), min_l2_indices]

        # Calculate regression loss between the best predicted trajectory and the ground truth
        # The `reduction='none'` in criterion means it returns loss per element/batch item
        # The sum over T,2 handles the per-element loss for the trajectory
        # Shape: (B, T, 2) -> sum to (B,)
        reg_loss_per_batch_item = F.mse_loss(
            best_pred_trajectory, ground_truth, reduction="none"
        )
        reg_loss_per_batch_item = torch.sum(
            reg_loss_per_batch_item, dim=[-1, -2]
        )  # Sum over T and 2

        # Combine the NLL probability loss and the regression loss
        total_loss_per_batch_item = (
            nll_prob_loss_per_batch_item + reg_loss_per_batch_item
        )

        mnll_loss = torch.mean(total_loss_per_batch_item)

        ### Combine the entropy loss and the MNLL loss ###
        loss = entropy_loss + mnll_loss

        return loss


# Example: A loss that penalizes sharp changes in predicted heading (if heading is part of prediction)
# class HeadingSmoothnessLoss(BaseLoss):
#     def __init__(self, name="HeadingSmoothnessLoss"):
#         super().__init__(name)
#
#     def forward(self, predictions, ground_truth, batch=None):
#         # Assuming predictions are (B, Seq_len, Dims) and one of the Dims is heading
#         # This is a placeholder for a more complex loss
#         if predictions.shape[-1] < 3: # Needs at least x, y, heading
#             return torch.tensor(0.0, device=predictions.device)
#
#         pred_headings = predictions[:, :, 2] # Example: 3rd dim is heading
#         heading_diffs = torch.diff(pred_headings, dim=1) # Difference between consecutive headings
#         return torch.mean(heading_diffs**2)


# --- Loss Registry and Getter ---
LOSS_REGISTRY = {
    config.Loss.MSETrajectoryLoss: MSETrajectoryLoss,
    config.Loss.L1TrajectoryLoss: L1TrajectoryLoss,
    config.Loss.GMMTrajectoryLoss: GMMTrajectoryLoss,  # Placeholder for GMM loss
    config.Loss.MNLLTrajectoryLoss: MNLLTrajectoryLoss,  # Placeholder for MNLL loss
    # "HeadingSmoothnessLoss": HeadingSmoothnessLoss,
}


def get_loss_instance(loss_name, **params):
    """
    Retrieves and instantiates a loss function from the registry.
    Args:
        loss_name (enum): The name of the loss function as registered.
        **params: Additional parameters to pass to the loss function's constructor.
                  The 'name' parameter is automatically handled if not provided in **params,
                  defaulting to loss_name.
    Returns:
        nn.Module: An instance of the requested loss function.
    Raises:
        ValueError: If the loss_name is not found in the registry.
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss function '{loss_name}' not found in registry. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
    loss_class = LOSS_REGISTRY[loss_name]

    # Ensure 'name' parameter is passed, defaulting to loss_name if not in params
    params["name"] = loss_name

    return loss_class(**params)
