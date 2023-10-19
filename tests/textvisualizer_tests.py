import unittest
from tidyX import TextVisualizer

class TestTextVisualizer(unittest.TestCase):

    def setUp(self):
        self.text_visualizer = TextVisualizer()

    def test_dependency_parse_valid_input(self):
        document = "This is a test sentence."
        # Testing if the function handles valid input correctly in non-Jupyter environment
        result = self.text_visualizer.dependency_parse_visualizer_text(document, jupyter=False)
        self.assertIsInstance(result, str)  # Expecting an HTML string as a result

    def test_dependency_parse_empty_input(self):
        document = ""
        # Testing if the function raises a ValueError for an empty string
        with self.assertRaises(ValueError):
            self.text_visualizer.dependency_parse_visualizer_text(document)

    def test_dependency_parse_invalid_style(self):
        document = "This is a test sentence."
        style = "invalid_style"
        # Testing if the function raises a ValueError for an invalid style
        with self.assertRaises(ValueError):
            self.text_visualizer.dependency_parse_visualizer_text(document, style=style)

    def test_dependency_parse_invalid_document_type(self):
        document = 123  # Non-string type
        # Testing if the function raises a ValueError for a non-string document
        with self.assertRaises(ValueError):
            self.text_visualizer.dependency_parse_visualizer_text(document)

    def test_dependency_parse_entity_visualization(self):
        document = "Microsoft Corporation is an American multinational technology company."
        style = "ent"
        # Testing entity visualization, expecting None as it is displayed directly in Jupyter
        result = self.text_visualizer.dependency_parse_visualizer_text(document, style=style)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
