import unittest
from tidyX import TextPreprocessor

class TestRemoveRepetitions(unittest.TestCase):
    
    def test_no_repetitions(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("hello"), "hello")
        
    def test_with_repetitions(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("heeello"), "hello")
        
    def test_with_exception_repetitions(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("heelloo"), "helloo")
        
    def test_with_mixed_repetitions(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("heeelloo"), "helloo")
        
    def test_empty_string(self):
        self.assertEqual(TextPreprocessor.remove_repetitions(""), "")
        
    def test_single_char_string(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("a"), "a")
        
    def test_single_repeated_char_string(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("aa"), "aa")
        
    def test_single_repeated_char_string_no_exception(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("bb"), "b")
        
    def test_multiple_repeated_char_string(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("aaaabbb"), "aab")
        
    def test_multiple_repeated_char_string_with_exception(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("aaaarrr"), "aarr")
        
if __name__ == '__main__':
    unittest.main()