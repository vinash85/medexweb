import unittest
from unittest.mock import MagicMock, patch, sys

# --- PRE-IMPORT MOCKING ---
# We mock llama_cpp BEFORE importing the server to avoid 
# dependencies like CUDA or GGUF files during logic testing.
mock_llama_mod = MagicMock()
sys.modules['llama_cpp'] = mock_llama_mod
sys.modules['llama_cpp.llama_chat_format'] = MagicMock()

# Now we can safely import the pipeline
from server import DermPipeline

class TestDermServer(unittest.TestCase):

    def setUp(self):
        """Setup the pipeline with mocked Llama instances."""
        # Patch the Llama class in the server module
        self.patcher = patch('server.Llama')
        self.mock_llama_class = self.patcher.start()
        
        # Initialize the pipeline
        # This will create two instances of mock_llama_class
        self.pipeline = DermPipeline()
        
        # Assign distinct mocks to vlm and reasoner for accurate call counting
        self.pipeline.vlm = MagicMock()
        self.pipeline.reasoner = MagicMock()

    def tearDown(self):
        self.patcher.stop()

    def test_pipeline_structure(self):
        """Verify the pipeline returns correct keys and counts calls accurately."""
        
        # 1. Define what the 'models' return
        mock_specialist_response = {
            "choices": [{"message": {"content": "Feature detected"}}]
        }
        mock_diagnosis_response = {
            "choices": [{"message": {"content": "MEL"}}]
        }
        
        self.pipeline.vlm.create_chat_completion.return_value = mock_specialist_response
        self.pipeline.reasoner.create_chat_completion.return_value = mock_diagnosis_response

        # 2. Mock os.path.exists so it thinks the image is there
        with patch('os.path.exists', return_value=True):
            result = self.pipeline.process("test_image.jpg")

        # 3. Assertions
        # Check dictionary keys
        self.assertIn("specialists", result)
        self.assertIn("diagnosis", result)
        
        # Check specialist findings count (should be 4)
        self.assertEqual(len(result["specialists"]), 4)
        self.assertEqual(result["diagnosis"], "MEL")
        
        # Check call counts
        # The Specialist (VLM) should be called exactly 4 times (one for each feature)
        self.assertEqual(self.pipeline.vlm.create_chat_completion.call_count, 4)
        
        # The Reasoner should be called exactly 1 time (final diagnosis)
        self.assertEqual(self.pipeline.reasoner.create_chat_completion.call_count, 1)

    def test_file_not_found(self):
        """Verify the pipeline handles missing files gracefully."""
        with patch('os.path.exists', return_value=False):
            result = self.pipeline.process("missing.jpg")
            
        self.assertIn("error", result)
        self.assertTrue(result["error"].startswith("File not found"))

if __name__ == '__main__':
    print("🧪 Running Synchronized Derm-MCP Unit Tests...")
    unittest.main()
