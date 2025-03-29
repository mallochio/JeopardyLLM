import os
import subprocess
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

def run_lora_finetuning(
    model_name: str = "google/gemma-3-1b-it",
    data_dir: str = "/Users/sid/Projects/code/JeopardyLLM/data/training_data",
    adapter_path: str = "/Users/sid/Projects/code/JeopardyLLM/models",
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    fine_tune_type: str = "lora",
    optimizer: str = "adamw",
    mask_prompt: bool = False,
    num_layers: int = 16,
    val_batches: int = -1,
    steps_per_report: int = 10,
    steps_per_eval: int = 100,
    resume_adapter_file: Optional[str] = None,
    save_every: int = 100,
    test: bool = False,
    test_batches: int = -1,
    max_seq_length: int = 2048,
    config_file: Optional[str] = None,
    grad_checkpoint: bool = False,
    seed: int = 42,
    additional_args: Optional[List[str]] = None
):
    """
    Run LoRA fine-tuning on the Gemma model using mlx_lm.
    
    Args:
        model_name: Path to the local model directory or Hugging Face repo
        data_dir: Directory with {train, valid, test}.jsonl files
        adapter_path: Save path for the fine-tuned weights
        iters: Iterations to train for
        batch_size: Minibatch size
        learning_rate: Learning rate
        fine_tune_type: Type of fine-tuning (lora, dora, or full)
        optimizer: Optimizer to use (adam or adamw)
        mask_prompt: Whether to mask the prompt in the loss
        num_layers: Number of layers to fine-tune (16 default, -1 for all)
        val_batches: Number of validation batches (-1 for entire set)
        steps_per_report: Number of steps between loss reporting
        steps_per_eval: Number of steps between validations
        resume_adapter_file: Path to resume training from
        save_every: Save model every N iterations
        test: Whether to evaluate on test set after training
        test_batches: Number of test set batches
        max_seq_length: Maximum sequence length
        config_file: Path to YAML configuration file
        grad_checkpoint: Whether to use gradient checkpointing
        seed: PRNG seed
        additional_args: Any additional arguments to pass
    """
    Path(adapter_path).mkdir(exist_ok=True, parents=True)
    
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--iters", str(iters),
        "--adapter-path", adapter_path,
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--fine-tune-type", fine_tune_type,
        "--optimizer", optimizer,
        "--num-layers", str(num_layers),
        "--val-batches", str(val_batches),
        "--steps-per-report", str(steps_per_report),
        "--steps-per-eval", str(steps_per_eval),
        "--save-every", str(save_every),
        "--max-seq-length", str(max_seq_length),
        "--seed", str(seed),
    ]
    
    if mask_prompt:
        cmd.append("--mask-prompt")
    
    if test:
        cmd.append("--test")
        cmd.append("--test-batches")
        cmd.append(str(test_batches))
    
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    
    if resume_adapter_file:
        cmd.extend(["--resume-adapter-file", resume_adapter_file])
    
    if config_file:
        cmd.extend(["-c", config_file])
    
    if additional_args:
        cmd.extend(additional_args)
    
    logger.info(f"Starting LoRA fine-tuning with command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        for line in iter(process.stdout.readline, ''):
            logger.info(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            logger.info("Fine-tuning completed successfully")
            return True
        else:
            logger.error(f"Fine-tuning failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logger.exception(f"Error during fine-tuning: {str(e)}")
        return False

def fuse_lora_weights(
    model_name="google/gemma-3-1b-it",
    adapter_file="jeopardy_adapters.npz",
    output_dir="gemma3_jeopardy_fused"
):
    """Merge LoRA weights with the base model."""
    cmd = [
        "python", "-m", "mlx_lm.fuse",
        "--model", model_name,
        "--adapter-file", adapter_file,
        "--out-dir", output_dir
    ]
    
    logger.info(f"Fusing LoRA weights with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Fusion completed: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Fusion failed: {e.stderr}")
        return False
