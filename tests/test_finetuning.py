import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import shutil
import sys

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.finetuning import run_lora_finetuning, fuse_lora_weights

class TestFinetuning(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        self.adapter_path = self.test_dir / "adapters"
        self.adapter_path.mkdir()
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('src.finetuning.subprocess.Popen')
    def test_run_lora_finetuning(self, mock_popen):
        """Test that run_lora_finetuning constructs the correct command and handles success."""
        # Setup mock
        process_mock = MagicMock()
        process_mock.stdout.readline.side_effect = ['Training started', 'Progress: 50%', 'Completed', '']
        process_mock.wait.return_value = None
        process_mock.returncode = 0
        mock_popen.return_value = process_mock
        
        # Execute the function
        result = run_lora_finetuning(
            model_name="test-model",
            data_dir="/test/data",
            adapter_path=str(self.adapter_path),
            iters=100,
            batch_size=2,
            learning_rate=2e-5,
            fine_tune_type="dora"
        )
        
        # Check results
        self.assertTrue(result)
        mock_popen.assert_called_once()
        
        # Verify command construction
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        self.assertIn("test-model", cmd)
        self.assertIn("100", cmd)
        self.assertIn("2", cmd)  # batch-size
        self.assertIn("2e-05", cmd)  # learning-rate
        self.assertIn("dora", cmd)  # fine_tune_type
        
    @patch('src.finetuning.subprocess.Popen')
    def test_run_lora_finetuning_failure(self, mock_popen):
        """Test that run_lora_finetuning handles subprocess failure."""
        # Setup mock for failure
        process_mock = MagicMock()
        process_mock.stdout.readline.side_effect = ['Training started', 'Error: Out of memory', '']
        process_mock.wait.return_value = None
        process_mock.returncode = 1
        mock_popen.return_value = process_mock
        
        # Execute the function
        result = run_lora_finetuning(
            model_name="test-model",
            data_dir="/test/data",
            adapter_path=str(self.adapter_path)
        )
        
        # Check results
        self.assertFalse(result)
    
    @patch('src.finetuning.subprocess.run')
    def test_fuse_lora_weights(self, mock_run):
        """Test that fuse_lora_weights constructs the correct command and handles success."""
        # Setup mock
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Fusion completed successfully"
        
        # Execute the function
        result = fuse_lora_weights(
            model_name="test-model",
            adapter_file="test_adapters.npz",
            output_dir="test_output"
        )
        
        # Check results
        self.assertTrue(result)
        mock_run.assert_called_once()
        
        # Verify command construction
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("test-model", cmd)
        self.assertIn("test_adapters.npz", cmd)
        self.assertIn("test_output", cmd)

if __name__ == '__main__':
    unittest.main()
