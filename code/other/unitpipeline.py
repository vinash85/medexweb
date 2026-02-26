import unittest
from unittest.mock import MagicMock, patch, sys
import os

# --- 1. MOCK LLAMA_CPP BEFORE IMPORT ---
# This prevents the test from trying to find CUDA drivers or GGUF files
mock_llama = MagicMock()
sys.modules['llama_cpp'] = mock_llama
sys.modules['llama_cpp.llama_chat_format'] = MagicMock()

# Now we import the pipeline class from your server file
from server import DermPipeline

class TestDermPipeline(unittest.TestCase):

    def setUp(self):
        """Set up the pipeline with mocked internal models."""
        # We patch the Llama class inside the server module
        with patch('server.Llama'):
            self.pipeline = DermPipeline()
            
            # Manually assign separate mocks to track call counts independently
            self.pipeline.vlm = MagicMock()
            self.pipeline.reasoner = MagicMock()

    def test_specialist_to_reasoner_flow(self):
        """Test if specialist findings are correctly aggregated for the reasoner."""
        
        # Define mock responses for the 4 specialist calls
        # We use side_effect to provide different text for each pass
        self.pipeline.vlm.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": "Network present"}}]},
            {"choices": [{"message": {"content": "No dots detected"}}]},
            {"choices": [{"message": {"content": "Blue-white veil seen"}}]},
            {"choices": [{"message": {"content": "Irregular borders" }}]}
        ]
        
        # Define mock response for the final diagnosis
        self.pipeline.reasoner.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "MEL"}}]
        }

        # Mock the file system check
        with patch('os.path.exists', return_value=True):
            result = self.pipeline.process("test_lesion.jpg")

        # --- ASSERTIONS ---
        
        # 1. Verify the Specialist (VLM) was called exactly 4 times
        self.assertEqual(self.pipeline.vlm.create_chat_completion.call_count, 4)
        
        # 2. Verify the Reasoner was called once with the aggregated context
        self.assertEqual(self.pipeline.reasoner.create_chat_completion.call_count, 1)
        
        # 3. Verify the final structure of the returned dictionary
        self.assertEqual(len(result["specialists"]), 4)
        self.assertEqual(result["diagnosis"], "MEL")
        self.assertIn("Blue-white veil seen", result["specialists"])

    def test_missing_image_logic(self):
        """Verify the pipeline handles missing files without attempting inference."""
        with patch('os.path.exists', return_value=False):
            result = self.pipeline.process("non_existent.jpg")
            
        self.assertIn("error", result)
        self.pipeline.vlm.create_chat_completion.assert_not_called()

if __name__ == '__main__':
    print("🧪 Testing DermPipeline Logic Flow...")
    unittest.main()
