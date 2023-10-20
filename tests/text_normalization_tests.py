import unittest
from tidyX import TextNormalization
import spacy

class TestTextNormalization(unittest.TestCase):

    def setUp(self):
        self.sample_documents = ["Hola, ¿Cómo estás?", "Muy bien, ¿y tú?", "Estoy bien, gracias."]
        self.nlp = spacy.load('es_core_news_sm')
        self.spacy_preprocessor = TextNormalization()

    def test_lemmatization(self):
        result = self.spacy_preprocessor.spanish_lemmatizer("corriendo", self.nlp)
        self.assertEqual(result, "correr")

        result = self.spacy_preprocessor.spanish_lemmatizer("está", self.nlp)
        self.assertEqual(result, "estar")

        result = self.spacy_preprocessor.spanish_lemmatizer("", self.nlp)
        self.assertEqual(result, "")

if __name__ == '__main__':
    unittest.main()
