# utils.py
import logging
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
import config


def get_console_logger():
    """Sets up the console logger."""
    logger = logging.getLogger("E2EDriver")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y/%m/%d %I:%M:%S %p",
        )
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger


logger = get_console_logger()


def setup_training_logger(log_dir, checkpoint_dir):
    """Creates directories for logging and checkpoints."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger.info(f"Tensorboard logs will be saved to: {log_dir}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    return writer


def save_checkpoint(state, is_best, checkpoint_dir, step, filename=None):
    """Saves model and training parameters."""
    state_keys = set(list(state.keys()))
    state_variables = config.CHECKPOINT_STATE_VARIABLES
    assert (
        state_variables == state_keys
    ), f"Checkpoint state keys mismatch. Expected: {state_variables}, Found: {state_keys}"

    epoch = state["epoch"]
    if filename is None:
        filename = (
            "epoch_" + str(epoch) + "_step_" + str(step) + "_" + "checkpoint.pth.tar"
        )
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"Saved new best model checkpoint to {best_filepath}")
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler):
    """Loads model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    logger.info(f"Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)

    state_keys = set(list(checkpoint.keys()))
    state_variables = config.CHECKPOINT_STATE_VARIABLES
    assert (
        state_variables == state_keys
    ), f"Checkpoint state keys mismatch. Expected: {state_variables}, Found: {state_keys}"

    # Adjust model state_dict keys if necessary (e.g., if saved with DataParallel)
    model_state_dict = checkpoint["state_dict"]
    model_new_state_dict = {}
    for k, v in model_state_dict.items():
        name = (
            k[7:] if k.startswith("module.") else k
        )  # remove `module.` prefix if present
        model_new_state_dict[name] = v

    model.load_state_dict(model_new_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_metric = checkpoint.get("best_metric", float("inf"))

    logger.info(
        f"Checkpoint loaded. Resuming from epoch {start_epoch}, global step {global_step}. Best metric so far: {best_metric:.4f}"
    )
    return start_epoch, global_step, best_metric


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    return total_size_bytes


def string_to_byte_array(text):
    """
    Converts a string to an array of integers representing individual bytes.

    Args:
        text: The input string.

    Returns:
        A list of integers, where each integer represents a byte.
    """
    byte_array = []
    for char in text:
        byte_array.append(ord(char))
    return byte_array


def byte_array_to_string(byte_array):
    """
    Converts an array of integers representing bytes back to a string.

    Args:
        byte_array: A list of integers, where each integer represents a byte.

    Returns:
        The decoded string.
    """
    return "".join(chr(b) for b in byte_array)


def sample_trajectory_from_gmm(coef, mu, ln_var):
    batch_size = mu.shape[0]
    # num_components = mu.shape[1]
    seq_len = mu.shape[2]
    dim = mu.shape[3]

    # Sample component index based on mixing coefficients
    coef = torch.softmax(coef, dim=1)  # Ensure coefficients sum to 1
    comp_idx = torch.argmax(
        coef, dim=1
    )  # Choose the component with the highest coefficient

    # Sample from component
    # Gather the means and log-variances for the selected component for each batch
    # comp_idx needs to be expanded to match the dimensions for gather
    comp_idx_expanded = comp_idx.reshape(batch_size, 1, 1, 1).expand(
        -1, -1, seq_len, dim
    )
    mean = torch.gather(mu, 1, comp_idx_expanded).squeeze(1)
    std_dev = torch.exp(torch.gather(ln_var, 1, comp_idx_expanded)).squeeze(1)
    sample = mean + std_dev * torch.randn_like(mean).to(config.DEVICE)

    return sample


def sample_trajectory_from_mnll(mode_probability, predicted_trajectory):
    batch_size = predicted_trajectory.shape[0]
    # num_components = predicted_trajectory.shape[1]
    seq_len = predicted_trajectory.shape[2]
    dim = predicted_trajectory.shape[3]

    # Determine top mode index
    mode_probability = torch.softmax(mode_probability, dim=1)
    mode_idx = torch.argmax(mode_probability, dim=1)

    # Sample from mode
    mode_idx_expanded = mode_idx.reshape(batch_size, 1, 1, 1).expand(
        -1, -1, seq_len, dim
    )
    sample = torch.gather(predicted_trajectory, 1, mode_idx_expanded).squeeze(1)

    return sample
