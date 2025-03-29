import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import (
    evaluate_factual_accuracy, 
    retrieve_similar_clues, 
    evaluate_jeopardy,
    evaluate_rag_enhanced_generation
)

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample validation data
        self.validation_data = [
            {"text": "<bos><start_of_turn>user\nCategory: HISTORY\nQuestion: This President\n<end_of_turn> <start_of_turn>model\nThe answer is: Abraham Lincoln<end_of_turn><eos>"},
            {"text": "<bos><start_of_turn>user\nCategory: SCIENCE\nQuestion: This element\n<end_of_turn> <start_of_turn>model\nThe answer is: Gold<end_of_turn><eos>"}
        ]
        self.validation_file = self.test_dir / "valid.jsonl"
        with open(self.validation_file, 'w') as f:
            for item in self.validation_data:
                f.write(json.dumps(item) + "\n")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_evaluate_factual_accuracy(self):
        """Test the factual accuracy evaluation function."""
        # Test exact match
        accuracy = evaluate_factual_accuracy("The capital of France is Paris.", "The capital of France is Paris.")
        self.assertEqual(accuracy, 1.0)
        
        # Test partial match
        accuracy = evaluate_factual_accuracy("The capital of France is Paris.", "Paris is the capital of France.")
        self.assertGreater(accuracy, 0)
        self.assertLess(accuracy, 1.0)
        
        # Test no match
        accuracy = evaluate_factual_accuracy("The capital of France is Paris.", "The capital of Italy is Rome.")
        self.assertLess(accuracy, 0.5)
        
        # Test empty reference
        accuracy = evaluate_factual_accuracy("Some text", "")
        self.assertEqual(accuracy, 0.0)
    
    def test_retrieve_similar_clues(self):
        """Test the retrieve_similar_clues function with and without RAG."""
        # Test without RAG
        clues = retrieve_similar_clues("Who was the first president?")
        self.assertIsInstance(clues, pd.DataFrame)
        self.assertEqual(len(clues), 2)  # Default is to return 2 fallback rows
        
        # Mock RAG instance
        mock_rag = MagicMock()
        mock_df = pd.DataFrame({
            'answer': ['George Washington was the first president', 'Washington served from 1789'],
            'category': ['PRESIDENTS', 'HISTORY'],
            'question': ['Who is George Washington?', 'Who is George Washington?']
        })
        mock_rag.retrieve_similar_clues.return_value = mock_df
        
        # Test with RAG
        clues = retrieve_similar_clues("Who was the first president?", rag_instance=mock_rag)
        self.assertIsInstance(clues, pd.DataFrame)
        self.assertEqual(len(clues), 2)
        self.assertIn('George Washington', clues['answer'].iloc[0])
    
    @patch('src.evaluation.test_generation')
    @patch('src.evaluation.load_model_for_evaluation')
    def test_evaluate_jeopardy(self, mock_load, mock_generate):
        """Test the Jeopardy evaluation function."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_generate.return_value = "This is Abraham Lincoln"
        
        # Run the evaluation
        results = evaluate_jeopardy(
            mock_model,
            mock_tokenizer,
            self.validation_file,
            n_samples=1
        )
        
        # Check the results
        self.assertIsNotNone(results)
        self.assertIn('results', results)
        self.assertIn('avg_factual_score', results)
        self.assertEqual(len(results['results']), 1)
        
        # Test with output file
        output_file = self.test_dir / "eval_results.json"
        results = evaluate_jeopardy(
            mock_model,
            mock_tokenizer,
            self.validation_file,
            n_samples=1,
            output_file=output_file
        )
        
        # Verify output file was created
        self.assertTrue(output_file.exists())
    
    @patch('src.evaluation.test_generation')
    def test_evaluate_rag_enhanced_generation(self, mock_generate):
        """Test the RAG-enhanced evaluation function."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_rag = MagicMock()
        mock_rag.generate_rag_prompt.return_value = "RAG-enhanced prompt"
        mock_rag.retrieve_similar_clues.return_value = pd.DataFrame({
            'answer': ['Similar clue'],
            'question': ['Similar response']
        })
        
        # Mock two different responses for standard vs RAG
        mock_generate.side_effect = ["Standard response", "RAG-enhanced response"]
        
        # Create test data with explicit columns
        test_df = pd.DataFrame({
            'answer': ['This president was the first to live in the White House'],
            'question': ['Who is John Adams?']
        })
        
        # Run the evaluation
        results = evaluate_rag_enhanced_generation(
            mock_model,
            mock_tokenizer,
            test_df,
            mock_rag,
            n_samples=1
        )
        
        # Check the results
        self.assertIsNotNone(results)
        self.assertIn('results', results)
        self.assertIn('summary', results)
        self.assertEqual(len(results['results']), 1)
        self.assertIn('standard_score', results['summary'])
        self.assertIn('rag_score', results['summary'])
        self.assertIn('avg_improvement', results['summary'])

if __name__ == '__main__':
    unittest.main()
