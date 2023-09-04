import unittest
from tidyX import TextPreprocessor
class TestTextPreprocessor(unittest.TestCase):

    def test_remove_repetitions(self):
        self.assertEqual(TextPreprocessor.remove_repetitions("hello"), "hello")
        self.assertEqual(TextPreprocessor.remove_repetitions("heeello"), "hello")
        self.assertEqual(TextPreprocessor.remove_repetitions("heelloo"), "helloo")
        self.assertEqual(TextPreprocessor.remove_repetitions("heeelloo"), "helloo")
        self.assertEqual(TextPreprocessor.remove_repetitions(""), "")
        self.assertEqual(TextPreprocessor.remove_repetitions("a"), "a")
        self.assertEqual(TextPreprocessor.remove_repetitions("aa"), "aa")
        self.assertEqual(TextPreprocessor.remove_repetitions("bb"), "b")
        self.assertEqual(TextPreprocessor.remove_repetitions("aaaabbb"), "aab")
        self.assertEqual(TextPreprocessor.remove_repetitions("aaaarrr"), "aarr")

    def test_remove_last_repetition(self):
        self.assertEqual(TextPreprocessor.remove_last_repetition("Holaaaa amigooo"), "Hola amigo")
        self.assertEqual(TextPreprocessor.remove_last_repetition("Testingggg itttt"), "Testing it")
        self.assertEqual(TextPreprocessor.remove_last_repetition("Amazinggg"), "Amazing")
        self.assertEqual(TextPreprocessor.remove_last_repetition(""), "")
        self.assertEqual(TextPreprocessor.remove_last_repetition("a"), "a")
        self.assertEqual(TextPreprocessor.remove_last_repetition("aa"), "aa")
        self.assertEqual(TextPreprocessor.remove_last_repetition("bb"), "bb")
        self.assertEqual(TextPreprocessor.remove_last_repetition("aaaabbb"), "aaaabbb")
        self.assertEqual(TextPreprocessor.remove_last_repetition("aaaarrr"), "aaaarrr")

    def test_remove_urls(self):
        self.assertEqual(TextPreprocessor.remove_urls("Visit our website at http://example.com"), "Visit our website at ")
        self.assertEqual(TextPreprocessor.remove_urls("Check out this link: https://www.example.com"), "Check out this link: ")
        self.assertEqual(TextPreprocessor.remove_urls("No URLs here."), "No URLs here.")
        self.assertEqual(TextPreprocessor.remove_urls(""), "")
        self.assertEqual(TextPreprocessor.remove_urls("a"), "a")
        self.assertEqual(TextPreprocessor.remove_urls("aa"), "aa")
        self.assertEqual(TextPreprocessor.remove_urls("bb"), "bb")
        self.assertEqual(TextPreprocessor.remove_urls("aaaabbb"), "aaaabbb")
        self.assertEqual(TextPreprocessor.remove_urls("aaaarrr"), "aaaarrr")
        
if __name__ == '__main__':
    unittest.main()
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