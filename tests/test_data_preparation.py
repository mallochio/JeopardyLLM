import unittest
import pandas as pd
import tempfile
from pathlib import Path
import shutil
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_preparation import generate_prompt, generate_knowledge_prompt, prepare_data

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.data_dir.mkdir()
        self.output_dir = self.test_dir / "output"
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'round': ['Jeopardy!', 'Double Jeopardy!'],
            'clue_value': ['$200', '$400'],
            'daily_double_value': [None, '$800'],
            'category': ['HISTORY', 'SCIENCE'],
            'comments': ['', 'Interesting fact'],
            'answer': ['This president was the first to live in the White House', 'This element has the symbol Au'],
            'question': ['Who is John Adams?', 'What is gold?'],
            'air_date': ['2020-01-01', '2020-01-02'],
            'notes': ['', '']
        })
        
        # Save sample data to a TSV file
        self.sample_data.to_csv(self.data_dir / "extra_matches.tsv", sep='\t', index=False)
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_generate_prompt(self):
        """Test the generate_prompt function formats data correctly."""
        row = self.sample_data.iloc[0]
        prompt = generate_prompt(row)
        
        # Check that the prompt contains key elements
        self.assertIn("<bos><start_of_turn>user", prompt)
        self.assertIn("You are Alex Trebek", prompt)
        self.assertIn("Jeopardy Round : Jeopardy!", prompt)
        self.assertIn("Category : HISTORY", prompt)
        self.assertIn('Here is your Jeopardy clue: "This president was the first to live in the White House"', prompt)
        self.assertIn("<end_of_turn><eos>", prompt)
    
    def test_generate_knowledge_prompt(self):
        """Test the generate_knowledge_prompt function formats data correctly."""
        row = self.sample_data.iloc[1]
        prompt = generate_knowledge_prompt(row)
        
        # Check that the prompt contains key elements
        self.assertIn("<bos><start_of_turn>user", prompt)
        self.assertIn("Topic: SCIENCE", prompt)
        self.assertIn("Question: This element has the symbol Au", prompt)
        self.assertIn("The answer is: What is gold?", prompt)
    
    def test_prepare_data_roleplay_style(self):
        """Test prepare_data with roleplay style produces the expected files."""
        train_df, valid_df = prepare_data(self.data_dir, self.output_dir, prompt_style="roleplay")
        
        # Check that output files were created
        train_path = self.output_dir / "train.jsonl"
        valid_path = self.output_dir / "valid.jsonl"
        self.assertTrue(train_path.exists())
        self.assertTrue(valid_path.exists())
        
        # Check train/validation split sizes (90% train, 10% valid)
        self.assertEqual(len(train_df) + len(valid_df), 2)
        
        # Make sure the right format was used (roleplay)
        with open(train_path, 'r') as f:
            content = f.read()
            self.assertIn("Alex Trebek", content)
    
    def test_prepare_data_knowledge_style(self):
        """Test prepare_data with knowledge style produces the expected files."""
        train_df, valid_df = prepare_data(self.data_dir, self.output_dir, prompt_style="knowledge")
        
        # Check that output files were created
        train_path = self.output_dir / "train.jsonl"
        valid_path = self.output_dir / "valid.jsonl"
        self.assertTrue(train_path.exists())
        self.assertTrue(valid_path.exists())
        
        # Make sure the right format was used (knowledge)
        with open(train_path, 'r') as f:
            content = f.read()
            self.assertIn("Topic:", content)
            self.assertIn("The answer is:", content)
            self.assertNotIn("Alex Trebek", content)

if __name__ == '__main__':
    unittest.main()
