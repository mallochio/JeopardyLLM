import os
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging(log_file="jeopardy_training.log", level=logging.INFO):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{log_file}"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured. Log file: {log_path}")
    return log_path

def save_training_config(config, output_path="training_config.json"):
    """Save training configuration to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training configuration saved to {output_path}")

def parse_training_logs(log_file):
    """Extract training metrics from log file."""
    metrics = {
        "loss": [],
        "iteration": []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "loss:" in line and "iter:" in line:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == "loss:":
                            try:
                                loss = float(parts[i+1])
                                metrics["loss"].append(loss)
                            except (ValueError, IndexError):
                                continue
                        if part == "iter:":
                            try:
                                iteration = int(parts[i+1])
                                metrics["iteration"].append(iteration)
                            except (ValueError, IndexError):
                                continue
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to parse training logs: {str(e)}")
        return metrics

def plot_training_metrics(metrics, output_path=None):
    """Plot training metrics."""
    if not metrics.get("loss") or not metrics.get("iteration"):
        logger.warning("No metrics data to plot")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["iteration"], metrics["loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Training metrics plot saved to {output_path}")
    else:
        plt.show()
