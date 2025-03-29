import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import faiss

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag import JeopardyRAG

class TestJeopardyRAG(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_path = self.test_dir / "data"
        self.data_path.mkdir()
        self.index_path = self.test_dir / "index"
        self.index_path.mkdir()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'round': ['Jeopardy!', 'Double Jeopardy!'],
            'category': ['HISTORY', 'SCIENCE'],
            'answer': ['This president was the first to live in the White House', 'This element has the symbol Au'],
            'question': ['Who is John Adams?', 'What is gold?']
        })
        self.sample_data.to_csv(self.data_path / "jeopardy_sample.tsv", sep='\t', index=False)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch('src.rag.SentenceTransformer')
    def test_init(self, mock_transformer):
        """Test RAG initialization."""
        # Setup mock
        mock_transformer.return_value = MagicMock()
        
        # Create RAG instance
        rag = JeopardyRAG(
            data_path=str(self.data_path),
            embedding_model_name="test-embeddings",
            index_path=str(self.index_path)
        )
        
        # Verify initialization
        self.assertEqual(rag.data_path, self.data_path)
        self.assertEqual(rag.index_path, self.index_path)
        self.assertEqual(rag.embedding_model_name, "test-embeddings")
        mock_transformer.assert_called_once_with("test-embeddings")
    
    @patch('src.rag.SentenceTransformer')
    def test_load_jeopardy_data(self, mock_transformer):
        """Test loading Jeopardy data from TSV files."""
        # Setup
        mock_transformer.return_value = MagicMock()
        rag = JeopardyRAG(data_path=str(self.data_path))
        
        # Load data
        df = rag.load_jeopardy_data()
        
        # Verify data loaded correctly
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn('answer', df.columns)
        self.assertIn('question', df.columns)
        self.assertEqual(df['answer'].iloc[0], 'This president was the first to live in the White House')
    
    @patch('src.rag.SentenceTransformer')
    @patch('src.rag.faiss.read_index')
    @patch('builtins.open', new_callable=mock_open)
    def test_build_vector_index_existing(self, mock_file, mock_read_index, mock_transformer):
        """Test loading existing vector index."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_read_index.return_value = mock_index
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps({
            "embedding_model": "test-model",
            "num_vectors": 2
        })
        
        # Create index file
        index_file = self.index_path / "jeopardy_faiss_index.bin"
        index_file.touch()
        metadata_file = self.index_path / "jeopardy_metadata.json"
        metadata_file.touch()
        
        # Initialize RAG and load index
        rag = JeopardyRAG(
            data_path=str(self.data_path),
            index_path=str(self.index_path)
        )
        
        # Load the index
        index = rag.build_vector_index(force_rebuild=False)
        
        # Verify index was loaded
        self.assertEqual(index, mock_index)
        mock_read_index.assert_called_once_with(str(index_file))
    
    @patch('src.rag.SentenceTransformer')
    @patch('src.rag.faiss.IndexFlatL2')
    @patch('src.rag.faiss.write_index')
    @patch('builtins.open', new_callable=mock_open)
    def test_build_vector_index_new(self, mock_file, mock_write_index, mock_index_flat, mock_transformer):
        """Test building a new vector index."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        mock_transformer.return_value.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index_flat.return_value = mock_index
        
        # Initialize RAG
        rag = JeopardyRAG(
            data_path=str(self.data_path),
            index_path=str(self.index_path)
        )
        
        # Force a new index build
        with patch('src.rag.tqdm') as mock_tqdm:
            index = rag.build_vector_index(force_rebuild=True)
        
        # Verify index was created
        self.assertEqual(index, mock_index)
        mock_index.add.assert_called_once()
        mock_write_index.assert_called_once()
    
    @patch('src.rag.SentenceTransformer')
    def test_retrieve_similar_clues(self, mock_transformer):
        """Test retrieving similar clues."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        
        # Create a mock index
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2]]),  # Distances
            np.array([[0, 1]])        # Indices
        )
        
        # Initialize RAG with mocked components
        rag = JeopardyRAG(data_path=str(self.data_path))
        rag.index = mock_index
        rag.clues_df = self.sample_data
        
        # Retrieve similar clues
        similar = rag.retrieve_similar_clues("Who was the first president?", k=2)
        
        # Verify results
        self.assertEqual(len(similar), 2)
        self.assertIn('distance', similar.columns)
        self.assertIn('category', similar.columns)
        self.assertIn('answer', similar.columns)
        self.assertIn('question', similar.columns)
    
    @patch('src.rag.SentenceTransformer')
    def test_generate_rag_prompt(self, mock_transformer):
        """Test generating RAG-enhanced prompt."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        
        # Initialize RAG with patched retrieve_similar_clues
        rag = JeopardyRAG(data_path=str(self.data_path))
        
        # Mock the retrieve_similar_clues method
        rag.retrieve_similar_clues = MagicMock()
        rag.retrieve_similar_clues.return_value = pd.DataFrame({
            'category': ['HISTORY', 'PRESIDENTS'],
            'answer': ['First president', 'White House resident'],
            'question': ['Who is Washington?', 'Who is Adams?']
        })
        
        # Generate RAG prompt
        prompt = rag.generate_rag_prompt("Who was the first president?")
        
        # Verify the prompt
        self.assertIsInstance(prompt, str)
        self.assertIn("<bos><start_of_turn>user", prompt)
        self.assertIn("Context", prompt)
        self.assertIn("Who is Washington?", prompt)
        self.assertIn("HISTORY", prompt)
        self.assertIn("Who was the first president?", prompt)

if __name__ == '__main__':
    unittest.main()
