import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path
import json
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, save_training_config, parse_training_logs, plot_training_metrics

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_basic_config, mock_stream_handler, mock_file_handler):
        """Test the logging setup function."""
        # Setup mocks
        log_dir = Path("logs")
        mock_file_handler.return_value = MagicMock()
        mock_stream_handler.return_value = MagicMock()
        
        # Run the function
        with patch('src.utils.Path.mkdir') as mock_mkdir:
            log_path = setup_logging("test_log.log")
        
        # Verify results
        mock_mkdir.assert_called_once()
        mock_basic_config.assert_called_once()
        self.assertIsNotNone(log_path)
        self.assertTrue("test_log.log" in str(log_path))
    
    def test_save_training_config(self):
        """Test saving training configuration."""
        # Create a test config
        config = {
            "model_name": "test-model",
            "batch_size": 4,
            "learning_rate": 1e-4
        }
        
        # Save the config
        output_path = self.test_dir / "config.json"
        save_training_config(config, output_path)
        
        # Verify the file was created with the right content
        self.assertTrue(output_path.exists())
        with open(output_path, 'r') as f:
            saved_config = json.load(f)
            self.assertEqual(saved_config, config)
    
    def test_parse_training_logs(self):
        """Test parsing training logs."""
        # Create a sample log file
        log_content = """
        2025-03-29 12:00:01 - INFO - Starting training...
        2025-03-29 12:00:02 - INFO - iter: 1, loss: 2.5
        2025-03-29 12:00:03 - INFO - iter: 2, loss: 2.3
        2025-03-29 12:00:04 - INFO - iter: 3, loss: 2.1
        2025-03-29 12:00:05 - INFO - Training complete
        """
        
        log_file = self.test_dir / "training.log"
        with open(log_file, 'w') as f:
            f.write(log_content)
        
        # Parse the logs
        metrics = parse_training_logs(log_file)
        
        # Verify results
        self.assertIn("loss", metrics)
        self.assertIn("iteration", metrics)
        self.assertEqual(len(metrics["loss"]), 3)
        self.assertEqual(len(metrics["iteration"]), 3)
        self.assertEqual(metrics["loss"], [2.5, 2.3, 2.1])
        self.assertEqual(metrics["iteration"], [1, 2, 3])
    
    def test_plot_training_metrics(self):
        """Test plotting training metrics."""
        # Create test metrics
        metrics = {
            "loss": [2.5, 2.3, 2.1, 1.9, 1.7],
            "iteration": [1, 2, 3, 4, 5]
        }
        
        # Plot and save to file
        output_path = self.test_dir / "plot.png"
        plot_training_metrics(metrics, output_path)
        
        # Verify the plot was saved
        self.assertTrue(output_path.exists())
    
    def test_plot_training_metrics_empty(self):
        """Test plotting with empty metrics."""
        # Create empty metrics
        metrics = {
            "loss": [],
            "iteration": []
        }
        
        # This should not create a file but also not error
        output_path = self.test_dir / "empty_plot.png"
        plot_training_metrics(metrics, output_path)
        
        # Verify no file was created
        self.assertFalse(output_path.exists())

if __name__ == '__main__':
    unittest.main()
