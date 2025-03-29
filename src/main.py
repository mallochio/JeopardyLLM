import argparse
import os
from pathlib import Path
import logging
import time

from data_preparation import prepare_data
from finetuning import run_lora_finetuning, fuse_lora_weights
from evaluation import (
    load_model_for_evaluation, 
    evaluate_jeopardy, 
    test_generation, 
    evaluate_rag_enhanced_generation
)
from utils import setup_logging, save_training_config, parse_training_logs, plot_training_metrics
from rag import JeopardyRAG

def main():
    parser = argparse.ArgumentParser(description="Jeopardy LLM Fine-tuning Pipeline")
    
    parser.add_argument("--data-dir", type=str, default="/Users/sid/Projects/code/JeopardyLLM/data", 
                        help="Directory containing the Jeopardy TSV files")
    parser.add_argument("--output-dir", type=str, default="/Users/sid/Projects/code/JeopardyLLM/models",
                        help="Directory to save models and outputs")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-1b-it",
                        help="Base model to fine-tune")
                        
    parser.add_argument("--fine-tune-type", type=str, choices=["lora", "dora", "full"], default="lora",
                        help="Type of fine-tuning to perform")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adamw",
                        help="Optimizer to use for training")
    parser.add_argument("--iters", type=int, default=600,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--mask-prompt", action="store_true",
                        help="Mask the prompt in the loss when training")
    parser.add_argument("--num-layers", type=int, default=16,
                        help="Number of layers to fine-tune (-1 for all)")
    parser.add_argument("--steps-per-report", type=int, default=10,
                        help="Number of steps between loss reporting")
    parser.add_argument("--steps-per-eval", type=int, default=100,
                        help="Number of steps between validations")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save model every N iterations")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Use gradient checkpointing to reduce memory use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--config-file", type=str, 
                        help="YAML configuration file with training options")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--run-test", action="store_true",
                        help="Evaluate on test set after training")
                        
    # Evaluation args
    parser.add_argument("--eval-samples", type=int, default=10,
                        help="Number of samples for evaluation")
                        
    # RAG args
    parser.add_argument("--use-rag", action="store_true",
                        help="Enable RAG functionality for enhanced factual accuracy")
    parser.add_argument("--build-rag-index", action="store_true",
                        help="Build RAG vector index from the Jeopardy data")
    parser.add_argument("--rag-embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding model to use for RAG")
    parser.add_argument("--rag-k", type=int, default=3,
                        help="Number of similar contexts to retrieve for RAG")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    training_data_dir = data_dir / "training_data"
    adapter_path = base_dir / "adapters"
    eval_output_path = base_dir / "eval_results"
    
    # Create directories
    base_dir.mkdir(exist_ok=True, parents=True)
    training_data_dir.mkdir(exist_ok=True, parents=True)
    adapter_path.mkdir(exist_ok=True, parents=True)
    eval_output_path.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(f"jeopardy_training_{timestamp}.log")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Jeopardy LLM fine-tuning pipeline with model {args.model_name}")
    
    # Save configuration
    config = vars(args)
    config["timestamp"] = timestamp
    save_training_config(config, base_dir / f"training_config_{timestamp}.json")
    
    try:
        # Step 1: Prepare data
        logger.info("Step 1: Preparing data")
        train_df, valid_df = prepare_data(
            data_dir,
            training_data_dir
        )
        
        # Step 2: Fine-tune model (unless skipped)
        if not args.skip_training:
            logger.info("Step 2: Running LoRA fine-tuning")
            success = run_lora_finetuning(
                model_name=args.model_name,
                data_dir=str(training_data_dir),
                adapter_path=str(adapter_path),
                iters=args.iters,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                fine_tune_type=args.fine_tune_type,
                optimizer=args.optimizer,
                mask_prompt=args.mask_prompt,
                num_layers=args.num_layers,
                steps_per_report=args.steps_per_report,
                steps_per_eval=args.steps_per_eval,
                save_every=args.save_every,
                test=args.run_test,
                max_seq_length=args.max_seq_length,
                config_file=args.config_file,
                grad_checkpoint=args.grad_checkpoint,
                seed=args.seed
            )
            
            if not success:
                logger.error("Fine-tuning failed. Exiting.")
                return
            
            # Parse training logs and plot metrics
            metrics = parse_training_logs(log_file)
            plot_path = base_dir / f"training_loss_{timestamp}.png"
            plot_training_metrics(metrics, output_path=plot_path)
            
            # Optionally, fuse LoRA weights (commented out for now)
            # adapter_file = list(adapter_path.glob("*.npz"))[0] if list(adapter_path.glob("*.npz")) else None
            # if adapter_file:
            #     fused_model_dir = base_dir / f"fused_model_{timestamp}"
            #     fuse_lora_weights(
            #         model_name=args.model_name,
            #         adapter_file=str(adapter_file),
            #         output_dir=str(fused_model_dir)
            #     )
        
        # Step 2.5: Initialize RAG if requested
        rag_instance = None
        if args.use_rag or args.build_rag_index:
            logger.info("Initializing RAG module")
            rag_instance = JeopardyRAG(
                data_path=str(data_dir),
                embedding_model_name=args.rag_embedding_model
            )
            
            if args.build_rag_index:
                logger.info("Building RAG vector index")
                rag_instance.build_vector_index(force_rebuild=True)
        
        # Step 3: Evaluate the model
        logger.info("Step 3: Evaluating the fine-tuned model")
        try:
            # Load model
            model, tokenizer = load_model_for_evaluation(
                args.model_name,
                adapter_path=str(adapter_path) if not args.skip_training else None
            )
            
            # Test basic generation
            logger.info("Running basic generation test...")
            test_prompt = "Hey! Are you Alex Trebek? Test me on Jeopardy!"
            response = test_generation(model, tokenizer, test_prompt)
            logger.info(f"Test response: {response}")
            
            # Run comprehensive evaluation
            valid_data_path = training_data_dir / "valid.jsonl"
            eval_results_file = eval_output_path / f"eval_results_{timestamp}.json"
            
            results = evaluate_jeopardy(
                model,
                tokenizer,
                valid_data_path,
                n_samples=args.eval_samples,
                output_file=eval_results_file
            )
            
            # Run RAG-enhanced evaluation if enabled
            if args.use_rag and rag_instance:
                logger.info("Running RAG-enhanced evaluation")
                rag_results_file = eval_output_path / f"rag_eval_results_{timestamp}.json"
                
                rag_results = evaluate_rag_enhanced_generation(
                    model,
                    tokenizer,
                    valid_data_path,
                    rag_instance,
                    n_samples=args.eval_samples,
                    output_file=rag_results_file
                )
                
                if rag_results:
                    improvement = rag_results["summary"]["avg_improvement"]
                    logger.info(f"RAG evaluation complete. Average improvement: {improvement:.4f}")
            
            if results:
                logger.info(f"Evaluation complete. Average factual score: {results['avg_factual_score']:.4f}")
            else:
                logger.error("Evaluation failed.")
                
        except Exception as e:
            logger.exception(f"Error during evaluation: {str(e)}")
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {str(e)}")

if __name__ == "__main__":
    main()
