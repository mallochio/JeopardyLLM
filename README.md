## 🏋🏻‍♀️ Note: This project is a work in progress. I'm going to be documenting here, adding more features and improving the code over time to improve usability.

# JeopardyLLM: An attempt at an AI-driven Jeopardy! Trainer 

This project implements a Retrieval-Augmented Generation (RAG) system using a LoRA fine-tuned gemma-3:1B model to create an AI-driven Jeopardy! trainer that focuses on factual accuracy and reduces hallucinations.

## Features
- **LoRA Fine-tuning**: Efficiently fine-tune the Gemma 3:1B model on M1 Macs using MLX
- **Retrieval-Augmented Generation**: Enhance factual accuracy by retrieving relevant historical clues
- **Custom Evaluation Metrics**: Track and measure factual consistency in generated content
- **Comprehensive Logging**: Monitor training progress and evaluation results

## Getting Started

### Prerequisites

- macOS with M1/M2/M3 chip
- Ollama installed with gemma3:1b model
- Python 3.9+

### Installation

1. Clone this repository
2. Install the requirements:

```bash
cd JeopardyLLM
pip install -r src/requirements.txt
```

### Data Preparation

The system uses Jeopardy! question data in TSV format with the following columns:
- round
- clue_value
- daily_double_value  
- category
- comments
- answer (the clue in Jeopardy! terminology)
- question (the response in Jeopardy! terminology)
- air_date
- notes

### Fine-tuning

To fine-tune the model:

```bash
python src/main.py --data-dir /path/to/data --output-dir /path/to/output --iters 1000
```

### Using RAG

To build the RAG vector index:

```bash
python src/main.py --skip-training --build-rag-index --data-dir /path/to/data
```

To use RAG for evaluation:

```bash
python src/main.py --skip-training --use-rag --eval-samples 20
```

### Interactive Demo

Try the interactive RAG demo to compare standard vs RAG-enhanced generation:

```bash
python src/rag_demo.py --adapter-path /path/to/adapters
```

Or compare using examples from a validation file:

```bash
python src/rag_demo.py --file data/training_data/valid.jsonl --samples 10 --output demo_results.json
```

## Project Structure

- `src/data_preparation.py`: Handles loading and formatting Jeopardy data
- `src/finetuning.py`: LoRA fine-tuning implementation for MLX
- `src/evaluation.py`: Custom metrics and evaluation functions
- `src/rag.py`: Retrieval-Augmented Generation implementation
- `src/utils.py`: Helper functions for logging and visualization
- `src/main.py`: End-to-end training and evaluation pipeline
- `src/rag_demo.py`: Interactive demo for comparing standard vs RAG generation

## RAG Architecture

The RAG system works by:
1. Indexing all Jeopardy clues in a FAISS vector database
2. For each new clue, finding semantically similar historical clues
3. Incorporating these similar clues as context in the prompt
4. Generating more factually accurate responses based on this context

This approach significantly reduces hallucinations and improves factual accuracy.

## Acknowledgments

- Data sourced from public Jeopardy! archives
- Built with MLX, the Apple accelerated machine learning framework