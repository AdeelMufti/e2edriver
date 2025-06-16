# metrics.py
import torch


def calculate_ade(predictions, ground_truth):
    """
    Calculates Average Displacement Error (ADE).
    Args:
        predictions (Tensor): Predicted trajectories (Batch, Seq_len, 2 or 3 for x,y,z).
        ground_truth (Tensor): Ground truth trajectories (Batch, Seq_len, 2 or 3).
    Returns:
        Tensor: Scalar ADE value for the batch.
    """
    # Ensure shapes match
    assert predictions.shape == ground_truth.shape
    assert predictions.dim() == 3 and predictions.shape[-1] >= 2  # B, T, (x,y,...)

    # Calculate Euclidean distance for each point in the trajectory
    # Use only x, y for calculation if more dimensions are present
    diff = predictions[..., :2] - ground_truth[..., :2]  # (B, T, 2)
    dist = torch.sqrt(torch.sum(diff**2, dim=-1))  # (B, T)

    # Average distance over the sequence length (time dimension)
    ade_per_sample = torch.mean(dist, dim=1)  # (B,)

    # Average ADE over the batch
    ade_batch = torch.mean(ade_per_sample)  # Scalar

    return ade_batch


# You can also calculate Final Displacement Error (FDE) easily:
def calculate_fde(predictions, ground_truth):
    """Calculates Final Displacement Error (FDE)."""
    assert predictions.shape == ground_truth.shape
    assert predictions.dim() == 3 and predictions.shape[-1] >= 2

    # Get the final points
    final_pred = predictions[:, -1, :2]  # (B, 2)
    final_gt = ground_truth[:, -1, :2]  # (B, 2)

    diff = final_pred - final_gt
    dist = torch.sqrt(torch.sum(diff**2, dim=-1))  # (B,)
    fde_batch = torch.mean(dist)  # Scalar
    return fde_batch
