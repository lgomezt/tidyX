import unittest
from tidyX import SpacyPreprocessor
import spacy

class TestSpacyPreprocessor(unittest.TestCase):

    def setUp(self):
        self.sample_documents = ["Hola, ¿Cómo estás?", "Muy bien, ¿y tú?", "Estoy bien, gracias."]
        self.nlp = spacy.load('es_core_news_sm')
        self.spacy_preprocessor = SpacyPreprocessor()

    def test_lemmatization(self):
        result = self.spacy_preprocessor.spanish_lemmatizer("corriendo", self.nlp)
        self.assertEqual(result, "correr")

        result = self.spacy_preprocessor.spanish_lemmatizer("está", self.nlp)
        self.assertEqual(result, "estar")

        result = self.spacy_preprocessor.spanish_lemmatizer("", self.nlp)
        self.assertEqual(result, "")

    def test_custom_lemmatizer(self):
        nlp = spacy.blank('es')
        lemmatizer = self.spacy_preprocessor.custom_lemmatizer(nlp, 'custom_lemmatizer')
        nlp.add_pipe(lemmatizer)
        doc = nlp("corriendo")
        self.assertEqual(doc[0].lemma_, "correr")

    def test_spacy_pipeline_default(self):
        processed_docs = self.spacy_preprocessor.spacy_pipeline(self.sample_documents)
        self.assertIsInstance(processed_docs, list)
        self.assertIsInstance(processed_docs[0], list)

    def test_spacy_pipeline_custom_lemmatizer(self):
        processed_docs = self.spacy_preprocessor.spacy_pipeline(self.sample_documents, custom_lemmatizer=True)
        self.assertIsInstance(processed_docs, list)
        self.assertIsInstance(processed_docs[0], list)

    def test_spacy_pipeline_pipeline_options(self):
        processed_docs = self.spacy_preprocessor.spacy_pipeline(self.sample_documents, pipeline=['tokenizer'])
        for doc in processed_docs:
            for token in doc:
                self.assertIsInstance(token, str)

    def test_spacy_pipeline_most_common_strings(self):
        processed_docs, most_common_words = self.spacy_preprocessor.spacy_pipeline(self.sample_documents, num_strings=2)
        self.assertIsInstance(most_common_words, list)
        self.assertEqual(len(most_common_words), 2)
        self.assertIsInstance(most_common_words[0], tuple)
        self.assertIsInstance(most_common_words[0][0], str)
        self.assertIsInstance(most_common_words[0][1], int)

if __name__ == '__main__':
    unittest.main()
