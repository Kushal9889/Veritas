import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Fix path so we can import 'src'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.orchestration.state import AgentState
from src.orchestration.nodes import retrieve_node, generate_node

class TestVeritasWorkflow(unittest.TestCase):

    # We patch the functions where they are USED (in src.orchestration.nodes), not where they are defined.
    @patch("src.orchestration.nodes.query_financial_docs")
    @patch("src.orchestration.nodes.query_graph")
    @patch("src.orchestration.nodes.extract_entities")
    def test_analyst_node(self, mock_extract, mock_graph, mock_vector):
        """
        Tests the Analyst Node by FAKING the database responses.
        No real keys required.
        """
        # 1. Setup the Mocks (The "Fake" Database)
        mock_extract.return_value = ["Tesla"]
        
        mock_vector.return_value = [
            {"text": "Tesla revenue is up 10%", "source": "tesla_10k.pdf", "page": 5},
            {"text": "Elon Musk is CEO", "source": "tesla_10k.pdf", "page": 1}
        ]
        
        mock_graph.return_value = [
             {"source": "Tesla", "relationship": "CEO_OF", "target": "Elon Musk"}
        ]
        
        # 2. Run the Node
        print("\n--- 🧪 TESTING ANALYST NODE (MOCKED) ---")
        initial_state = AgentState(
            question="What is the revenue?",
            documents=[],
            graph_data=[],
            generation="",
            grade="",
            retry_count=0
        )
        
        # This will call our fake functions instead of the real ones
        result = retrieve_node(initial_state)
        
        # 3. Assertions (Did the logic work?)
        # Check if it formatted the document string correctly
        self.assertTrue(len(result["documents"]) > 0)
        self.assertIn("(Source: tesla_10k.pdf, Page 5)", result["documents"][0])
        print("✅ Analyst Node returned documents correctly")

    @patch("src.orchestration.nodes.genai.GenerativeModel")
    def test_writer_node(self, mock_model_class):
        """
        Tests the Writer Node by FAKING the Gemini AI.
        """
        # 1. Setup the Mock (The "Fake" AI)
        mock_instance = mock_model_class.return_value
        # When .generate_content() is called, return this object with a .text attribute
        mock_instance.generate_content.return_value.text = "Tesla revenue increased by 10%."
        
        # 2. Run the Node
        print("\n--- 🧪 TESTING WRITER NODE (MOCKED) ---")
        state = AgentState(
            question="What is the revenue?",
            documents=["Content: Tesla revenue is up. (Source: tesla_10k.pdf, Page 5)"],
            graph_data=[],
            generation="",
            grade="",
            retry_count=0
        )
        
        result = generate_node(state)
        
        # 3. Assertions
        self.assertIn("Tesla", result["generation"])
        print("✅ Writer Node generated text correctly")

if __name__ == "__main__":
    unittest.main()