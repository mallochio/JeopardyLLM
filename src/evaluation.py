import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from mlx_lm import generate, load
import json
from typing import List, Dict, Any, Optional
from rag import JeopardyRAG
import re
import spacy
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_md")
    logger.info("Loaded spaCy model for evaluation metrics")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    logger.warning("Using simplified metrics without NLP capabilities")
    nlp = None

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
        
        # Generate visualization for factual accuracy
        visualize_factual_accuracy(results, avg_factual_score, output_file)

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
        
        # Generate visualization for RAG evaluation
        visualize_rag_results(results, summary, output_file)

        return {
            "results": results,
            "summary": summary
        }
            
    except Exception as e:
        logger.exception(f"RAG evaluation failed: {str(e)}")
        return None


def visualize_factual_accuracy(results, avg_factual_score, output_file=None):
    """
    Visualize factual accuracy results as a bar chart.
    """
    prompts = [result["prompt"][:30] + "..." for result in results]
    scores = [result["factual_score"] for result in results]

    plt.figure(figsize=(10, 6))
    plt.barh(prompts, scores, color="skyblue")
    plt.axvline(avg_factual_score, color="red", linestyle="--", label="Average Score")
    plt.xlabel("Factual Accuracy Score")
    plt.ylabel("Sample Prompts")
    plt.title("Factual Accuracy Evaluation")
    plt.legend()
    plt.tight_layout()

    if output_file:
        plot_path = output_file.replace(".json", "_factual_accuracy.png")
        plt.savefig(plot_path)
        logging.info(f"Factual accuracy plot saved to {plot_path}")
    else:
        plt.show()


def visualize_rag_results(results, summary, output_file=None):
    """
    Visualize RAG evaluation results as a comparison bar chart.
    """
    clues = [result["clue"][:30] + "..." for result in results]
    standard_scores = [result["standard_score"] for result in results]
    rag_scores = [result["rag_score"] for result in results]

    x = np.arange(len(clues))
    width = 0.35

    plt.figure(figsize=(12, 7))
    plt.bar(x - width / 2, standard_scores, width, label="Standard")
    plt.bar(x + width / 2, rag_scores, width, label="RAG-Enhanced", color="orange")
    plt.xlabel("Clues")
    plt.ylabel("Scores")
    plt.title("RAG vs Standard Evaluation")
    plt.xticks(x, clues, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    if output_file:
        plot_path = output_file.replace(".json", "_rag_comparison.png")
        plt.savefig(plot_path)
        logging.info(f"RAG comparison plot saved to {plot_path}")
    else:
        plt.show()


class JeopardyEvaluator:
    """Evaluates factual accuracy and hallucination reduction in Jeopardy answers"""
    
    def __init__(self, use_rag: bool = True):
        """
        Initialize the evaluator
        
        Args:
            use_rag: Whether to use RAG-specific metrics
        """
        self.use_rag = use_rag
        self.metrics = {}
    
    def evaluate(self, 
                 generated_answers: List[str], 
                 correct_answers: List[str], 
                 retrieved_contexts: List[List[str]] = None,
                 base_model_answers: List[str] = None) -> Dict[str, float]:
        """
        Run all evaluation metrics
        
        Args:
            generated_answers: List of model-generated answers
            correct_answers: List of ground truth answers
            retrieved_contexts: List of retrieved contexts used for each answer
            base_model_answers: List of answers from base model without RAG
            
        Returns:
            Dictionary of metric names and scores
        """
        if len(generated_answers) != len(correct_answers):
            raise ValueError("Number of generated and correct answers must match")
            
        # Initialize results dictionary
        results = {}
        
        # Calculate Answer Accuracy Score
        acc_scores = []
        for gen, correct in zip(generated_answers, correct_answers):
            acc_scores.append(calculate_answer_accuracy(gen, correct, nlp))
        results["answer_accuracy_score"] = np.mean(acc_scores)
        
        # Entity Consistency Score
        if retrieved_contexts:
            ecs_scores = []
            for gen, contexts in zip(generated_answers, retrieved_contexts):
                ecs_scores.append(calculate_entity_consistency(gen, contexts, nlp))
            results["entity_consistency_score"] = np.mean(ecs_scores)
        
        # Hallucination Detection Score
        if retrieved_contexts:
            hds_scores = []
            for gen, contexts in zip(generated_answers, retrieved_contexts):
                hds_scores.append(detect_hallucinations(gen, contexts))
            results["hallucination_score"] = np.mean(hds_scores)
        
        # RAG Enhancement Metric
        if self.use_rag and base_model_answers:
            results["rag_enhancement"] = measure_rag_enhancement(
                base_model_answers, generated_answers, correct_answers, nlp)
        
        # Log the results
        logger.info(f"Evaluation results: {results}")
        self.metrics = results
        return results
    
    def get_summary(self) -> str:
        """Generate a human-readable summary of evaluation results"""
        if not self.metrics:
            return "No evaluation has been performed yet."
            
        summary = "=== Jeopardy LLM Evaluation Results ===\n"
        
        if "answer_accuracy_score" in self.metrics:
            summary += f"Answer Accuracy: {self.metrics['answer_accuracy_score']:.2f}\n"
            
        if "entity_consistency_score" in self.metrics:
            summary += f"Entity Consistency: {self.metrics['entity_consistency_score']:.2f}\n"
            
        if "hallucination_score" in self.metrics:
            summary += f"Hallucination Score: {self.metrics['hallucination_score']:.2f} (lower is better)\n"
            
        if "rag_enhancement" in self.metrics:
            rag_effect = "helped" if self.metrics['rag_enhancement'] > 0 else "hurt"
            summary += f"RAG Enhancement: {self.metrics['rag_enhancement']:.2f} (RAG {rag_effect})\n"
            
        return summary
    
def detect_hallucinations(generated_answer, retrieved_contexts):
    """
    Identifies potential numerical, date, or fact hallucinations.
    
    Args:
        generated_answer (str): The model's generated answer
        retrieved_contexts (list): List of retrieved passages used as context
        
    Returns:
        float: Hallucination likelihood score (lower is better)
    """
    # Extract numbers and dates from the generated answer
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', generated_answer)
    years = re.findall(r'\b(19|20)\d{2}\b', generated_answer)
    dates = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', generated_answer)
    
    total_facts = len(numbers) + len(years) + len(dates)
    if total_facts == 0:
        return 0.0  # No numeric facts to verify
    
    # Check if each number/date appears in any context
    verified_facts = 0
    for fact in numbers + years + dates:
        if any(fact in context for context in retrieved_contexts):
            verified_facts += 1
    
    # Return hallucination score (percentage of unverified facts)
    return 1.0 - (verified_facts / total_facts) if total_facts > 0 else 0.0

def calculate_entity_consistency(generated_answer, retrieved_contexts, nlp_model):
    """
    Check if entities mentioned in the generated answer appear in retrieved contexts.
    
    Args:
        generated_answer (str): The model's generated answer
        retrieved_contexts (list): List of retrieved passages used as context
        nlp_model: A spaCy model with NER capability
        
    Returns:
        float: Consistency score between 0-1
    """
    # Extract named entities from the generated answer
    gen_doc = nlp_model(generated_answer)
    generated_entities = set([ent.text.lower() for ent in gen_doc.ents])
    
    if not generated_entities:
        return 1.0  # No entities to verify
    
    # Extract entities from contexts
    context_entities = set()
    for context in retrieved_contexts:
        context_doc = nlp_model(context)
        context_entities.update([ent.text.lower() for ent in context_doc.ents])
    
    # Calculate what percentage of generated entities appear in contexts
    if generated_entities:
        matched = sum(1 for entity in generated_entities if any(
            entity in context.lower() for context in retrieved_contexts))
        return matched / len(generated_entities)
    
    return 0.0

def calculate_answer_accuracy(generated_answer, correct_answer, nlp_model):
    """
    Measures semantic similarity between generated answer and correct answer.
    
    Args:
        generated_answer (str): The model's generated answer
        correct_answer (str): The ground truth answer
        nlp_model: A spaCy or similar NLP model for embeddings
        
    Returns:
        float: Similarity score between 0-1
    """
    # Normalize answers (remove "what is", "who is", etc.)
    gen_normalized = re.sub(r'^(what|who|where|when|how) (is|are|was|were) ', '', 
                           generated_answer.lower().strip())
    correct_normalized = re.sub(r'^(what|who|where|when|how) (is|are|was|were) ', '', 
                               correct_answer.lower().strip())
    
    # Get document vectors
    gen_doc = nlp_model(gen_normalized)
    correct_doc = nlp_model(correct_normalized)
    
    # Calculate cosine similarity
    if gen_doc.vector_norm and correct_doc.vector_norm:
        return gen_doc.similarity(correct_doc)
    return 0.0


def measure_rag_enhancement(base_model_answers, rag_model_answers, correct_answers, nlp_model):
    """
    Quantifies how much RAG improves answer accuracy compared to the base model.
    
    Args:
        base_model_answers (list): Answers from the base model without RAG
        rag_model_answers (list): Answers from the model with RAG
        correct_answers (list): Ground truth answers
        nlp_model: NLP model for semantic similarity
        
    Returns:
        float: Average improvement from using RAG (-1 to 1)
    """
    improvements = []
    
    for base, rag, correct in zip(base_model_answers, rag_model_answers, correct_answers):
        # Calculate similarity scores
        base_similarity = calculate_answer_accuracy(base, correct, nlp_model)
        rag_similarity = calculate_answer_accuracy(rag, correct, nlp_model)
        
        # Improvement is difference in similarities
        improvements.append(rag_similarity - base_similarity)
    
    # Return average improvement (positive means RAG helped)
    return sum(improvements) / len(improvements) if improvements else 0.0

