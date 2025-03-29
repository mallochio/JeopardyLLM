import numpy as np
import pandas as pd
import logging
from pathlib import Path
from mlx_lm import generate, load
import json
from typing import List, Dict, Any, Optional
from rag import JeopardyRAG

logger = logging.getLogger(__name__)

def load_model_for_evaluation(model_name, adapter_path=None):
    """Load the model for evaluation."""
    logger.info(f"Loading model {model_name} {'with adapter' if adapter_path else 'without adapter'}")
    try:
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def test_generation(model, tokenizer, prompt, max_tokens=6000):
    """Test basic generation with the model."""
    logger.info(f"Testing generation with prompt: {prompt[:50]}...")
    
    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=True
        )
        return response
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return None

def retrieve_similar_clues(clue, k=2, rag_instance=None):
    """
    Retrieve similar clues using the RAG module if available,
    otherwise return dummy data.
    """
    if rag_instance:
        return rag_instance.retrieve_similar_clues(clue, k=k)
    else:
        return pd.DataFrame({
            'answer': [f"Similar clue 1 to {clue}", f"Similar clue 2 to {clue}"]
        })

def evaluate_factual_accuracy(generated: str, reference: str) -> float:
    """
    Simple evaluation of factual accuracy.
    This could be replaced with a more sophisticated metric.
    """
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    overlap = len(gen_words.intersection(ref_words))
    return overlap / len(ref_words)

def evaluate_jeopardy(model, tokenizer, validation_data_path, n_samples=10, output_file=None):
    """
    Evaluate the model on Jeopardy clues.
    """
    logger.info(f"Evaluating model on {n_samples} samples from {validation_data_path}")
    
    try:
        with open(validation_data_path, 'r') as f:
            val_data = [json.loads(line) for line in f]
        
        if len(val_data) < n_samples:
            n_samples = len(val_data)
            logger.warning(f"Reduced sample size to {n_samples} due to available data")
            
        sample_indices = np.random.choice(len(val_data), size=n_samples, replace=False)
        
        results = []
        total_factual_score = 0.0
        
        for idx in sample_indices:
            sample = val_data[idx]
            original_text = sample['text']
            prompt_parts = original_text.split("<start_of_turn>model")
            
            if len(prompt_parts) > 1:
                user_prompt = prompt_parts[0] + "<start_of_turn>model"
                expected = prompt_parts[1].strip()
            else:
                user_prompt = original_text
                expected = ""
            
            generated = test_generation(model, tokenizer, user_prompt, max_tokens=100)
            
            factual_score = evaluate_factual_accuracy(generated, expected)
            total_factual_score += factual_score
            
            results.append({
                "prompt": user_prompt,
                "expected": expected,
                "generated": generated,
                "factual_score": factual_score
            })
        
        avg_factual_score = total_factual_score / n_samples
        logger.info(f"Average factual accuracy: {avg_factual_score:.4f}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "results": results,
                    "avg_factual_score": avg_factual_score
                }, f, indent=2)
            logger.info(f"Evaluation results saved to {output_file}")
        
        return {
            "results": results,
            "avg_factual_score": avg_factual_score
        }
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return None

def evaluate_rag_enhanced_generation(model, tokenizer, data_path, rag_instance, n_samples=10, output_file=None):
    """
    Evaluate the model with RAG-enhanced prompts.
    Compare standard generation vs RAG-enhanced generation.
    """
    logger.info(f"Evaluating RAG-enhanced generation on {n_samples} samples")
    
    try:
        if isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            data_path = Path(data_path)
            if data_path.suffix == '.jsonl':
                with open(data_path, 'r') as f:
                    val_data = [json.loads(line) for line in f]
                    if val_data and 'text' in val_data[0]:
                        df = pd.DataFrame(val_data)
                    else:
                        logger.error("Invalid data format")
                        return None
            else:
                df = pd.read_csv(data_path, sep="\t")
        
        if len(df) > n_samples:
            df_sample = df.sample(n=n_samples, random_state=42)
        else:
            df_sample = df
            logger.warning(f"Sample size reduced to {len(df)} due to available data")
        
        results = []
        
        for _, row in df_sample.iterrows():
            if 'text' in row:
                text = row['text']
                try:
                    clue_start = text.find('Here is your Jeopardy clue: "')
                    if clue_start != -1:
                        clue_start += len('Here is your Jeopardy clue: "')
                        clue_end = text.find('"', clue_start)
                        clue = text[clue_start:clue_end]
                    else:
                        clue = "Sample clue"
                        
                    question_marker = "Question : "
                    question_start = text.find(question_marker)
                    if question_start != -1:
                        question_start += len(question_marker)
                        question_end = text.find("\n", question_start)
                        if question_end == -1:
                            question_end = len(text)
                        expected_question = text[question_start:question_end]
                    else:
                        expected_question = ""
                except Exception as e:
                    logger.warning(f"Error parsing text: {e}")
                    clue = "Sample clue"
                    expected_question = ""
            else:
                clue = row.get('answer', 'Sample clue')
                expected_question = row.get('question', '')
            
            standard_prompt = f"<bos><start_of_turn>user\nYou are Alex Trebek. Present this Jeopardy clue: \"{clue}\"\n<end_of_turn> <start_of_turn>model"
            standard_response = test_generation(model, tokenizer, standard_prompt, max_tokens=100)
            
            rag_prompt = rag_instance.generate_rag_prompt(clue, k=3)
            rag_response = test_generation(model, tokenizer, rag_prompt, max_tokens=100)
            
            standard_score = evaluate_factual_accuracy(standard_response, expected_question)
            rag_score = evaluate_factual_accuracy(rag_response, expected_question)
            
            results.append({
                "clue": clue,
                "expected_question": expected_question,
                "standard_response": standard_response,
                "rag_response": rag_response,
                "standard_score": standard_score,
                "rag_score": rag_score,
                "improvement": rag_score - standard_score
            })
        
        avg_standard = np.mean([r["standard_score"] for r in results])
        avg_rag = np.mean([r["rag_score"] for r in results])
        avg_improvement = np.mean([r["improvement"] for r in results])
        
        summary = {
            "avg_standard_score": float(avg_standard),
            "avg_rag_score": float(avg_rag),
            "avg_improvement": float(avg_improvement),
            "num_samples": len(results)
        }
        
        logger.info(f"RAG Evaluation Results:")
        logger.info(f"  Average standard score: {avg_standard:.4f}")
        logger.info(f"  Average RAG score: {avg_rag:.4f}")
        logger.info(f"  Average improvement: {avg_improvement:.4f}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "results": results,
                    "summary": summary
                }, f, indent=2)
            logger.info(f"RAG evaluation results saved to {output_file}")
        
        return {
            "results": results,
            "summary": summary
        }
            
    except Exception as e:
        logger.exception(f"RAG evaluation failed: {str(e)}")
        return None
