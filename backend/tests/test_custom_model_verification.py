import unittest
from unittest.mock import patch, AsyncMock
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.custom_model import query_custom_model
from backend.openrouter import query_model


class TestCustomModel(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Hello"}]

    @patch('backend.custom_model.httpx.AsyncClient')
    def test_query_custom_model_success(self, mock_client):
        from unittest.mock import MagicMock
        # Setup mock response - json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Hello from local model',
                    'reasoning_details': None
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        
        # Setup client - post() is async
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        
        # Setup context manager
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Run test
        result = asyncio.run(query_custom_model("local/test-model", self.messages))
        
        self.assertIsNotNone(result)
        self.assertEqual(result['content'], 'Hello from local model')

    @patch('backend.openrouter.query_custom_model', new_callable=AsyncMock)
    def test_openrouter_routing(self, mock_query_custom):
        # Setup mock - AsyncMock properly handles awaiting
        mock_query_custom.return_value = {'content': 'Routed correctly', 'reasoning_details': None}
        
        # Test routing
        result = asyncio.run(query_model("local/my-model", self.messages))
        
        mock_query_custom.assert_called_once()
        self.assertEqual(result['content'], 'Routed correctly')

if __name__ == '__main__':
    unittest.main()
