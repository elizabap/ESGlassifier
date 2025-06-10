from transformers import TrainerCallback
import torch
import wandb

class MemoryLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
            wandb.log({"gpu_memory_mb": memory}, step=state.global_step)
            torch.cuda.reset_peak_memory_stats()