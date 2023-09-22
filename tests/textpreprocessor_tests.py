import unittest
import sys
from tidyX import TextPreprocessor as tp
import unidecode
import pandas as pd
import spacy
import numpy as np
from nltk.corpus import stopwords

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.sample_tweets = [
            "¡Hola! ¿Cómo estás? 😀 #buenasvibes",
            "Me encantó este libro 📚👏👏",
            "¡Increíble! 😲 No puedo creerlo... https://example.com",
            "Amo los días soleados ☀️, pero también la lluvia 🌧️.",
            "🤔 Pienso, luego existo. #filosofía",
            "Ahoraaaa, me encantaaaaas masssss! Que rico bb! 😍��",
            "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' !! 😏"
        ]
        self.sample_documents = ["Hola, ¿Cómo estás?", "Muy bien, ¿y tú?", "Estoy bien, gracias."]
        self.nlp = spacy.load('es_core_news_sm')
        self.text_preprocessor = TextPreprocessor()


    def test_remove_repetitions(self):
        # Testing first tweet: "¡Hola! ¿Cómo estás? 😀 #buenasvibes"
        processed = tp.remove_repetitions(self.sample_tweets[0])
        self.assertEqual(processed, "¡Hola! ¿Cómo estás? 😀 #buenasvibes")
        
        # Testing second tweet: "Me encantó este libro 📚👏👏"
        processed = tp.remove_repetitions(self.sample_tweets[1])
        self.assertEqual(processed, "Me encantó este libro 📚👏")
        
        # Testing sixth tweet: "Ahoraaaa, me encantaaaaas masssss! Que rico bb! 😍��"
        processed = tp.remove_repetitions(self.sample_tweets[5])
        self.assertEqual(processed, "Ahoraaa, me encantaaaas masss! Que rico b! 😍��")

        # Testing seventh tweet: "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' !! 😏"
        processed = tp.remove_repetitions(self.sample_tweets[6])
        self.assertEqual(processed, "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' ! 😏")

        # Test with custom exceptions
        processed = tp.remove_repetitions(self.sample_tweets[5], exceptions=["a", "s"])
        self.assertEqual(processed, "Ahoraaa, me encantaaaas masss! Que rico b! 😍��")
    
    def test_remove_last_repetition(self):
        # Testing the sixth tweet: "Ahoraaaa, me encantaaaaas masssss! Que rico bb! 😍��"
        processed = tp.remove_last_repetition(self.sample_tweets[5])
        self.assertEqual(processed, "Ahora, me encantas mas! Que rico b! 😍��")

        # Testing the seventh tweet: "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' !! 😏"
        processed = tp.remove_last_repetition(self.sample_tweets[6])
        self.assertEqual(processed, "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' ! 😏")

        # Testing a sentence where each word ends with the last character repeated: "Holaaaa amigooo"
        processed = tp.remove_last_repetition("Holaaaa amigooo")
        self.assertEqual(processed, "Hola amigo")

        # Testing a sentence where each word ends with the last character repeated multiple times: "Holaaaaaa amigooooo"
        processed = tp.remove_last_repetition("Holaaaaaa amigooooo")
        self.assertEqual(processed, "Hola amigo")

        # Testing sentence without any repetition at the end of words
        processed = tp.remove_last_repetition("This is a normal sentence.")
        self.assertEqual(processed, "This is a normal sentence.")

    def test_remove_urls(self):
        # Testing the third tweet: "¡Increíble! 😲 No puedo creerlo... https://example.com"
        processed = tp.remove_urls(self.sample_tweets[2])
        self.assertEqual(processed, "¡Increíble! 😲 No puedo creerlo... ")

        # Testing sentence with multiple URLs
        processed = tp.remove_urls("Visit http://example1.com and https://example2.com")
        self.assertEqual(processed, "Visit  and ")

        # Testing sentence without any URLs
        processed = tp.remove_urls("This is a normal sentence.")
        self.assertEqual(processed, "This is a normal sentence.")
        
        # Testing sentence with an incomplete URL
        processed = tp.remove_urls("Visit http")
        self.assertEqual(processed, "Visit ")

        # Testing sentence with URL followed by a special character
        processed = tp.remove_urls("Visit https://example.com.")
        self.assertEqual(processed, "Visit .")

    def test_remove_RT(self):

        processed = tp.remove_RT(self.sample_tweets[1])
        self.assertEqual(processed, "Me encantó este libro 📚👏👏")

        # Testing a tweet with "RT" inside but not as prefix: "Nice RT! This is awesome."
        processed = tp.remove_RT("Nice RT! This is awesome.")
        self.assertEqual(processed, "Nice RT! This is awesome.")    

    def test_remove_accents(self):
        # Test with delete_emojis = True (Default)
        processed = tp.remove_accents(self.sample_tweets[0])
        self.assertEqual(processed, unidecode(self.sample_tweets[0]))

        # Test with delete_emojis = False
        processed = tp.remove_accents(self.sample_tweets[0], delete_emojis=False)
        self.assertEqual(processed, "¡Hola! ¿Como estas? 😀 #buenasvibes")

        # Test with accented characters but no emojis
        processed = tp.remove_accents("Me encantó este libro", delete_emojis=False)
        self.assertEqual(processed, "Me encanto este libro")

        # Test without accented characters but with emojis
        processed = tp.remove_accents("Awesome 😀👏👏", delete_emojis=True)
        self.assertEqual(processed, "Awesome ")

        # Test with neither accented characters nor emojis
        processed = tp.remove_accents("Awesome", delete_emojis=True)
        self.assertEqual(processed, "Awesome")

    def test_remove_hashtags(self):
        # Test tweet with a single hashtag
        processed = tp.remove_hashtags(self.sample_tweets[4])
        self.assertEqual(processed, "🤔 Pienso, luego existo. ")

        # Test tweet with multiple hashtags
        processed = tp.remove_hashtags("This is #amazing! #awesome #fantastic")
        self.assertEqual(processed, "This is !  ")

        # Test tweet with no hashtags
        processed = tp.remove_hashtags("This is amazing!")
        self.assertEqual(processed, "This is amazing!")

        # Test tweet with hashtag embedded in text (should be removed)
        processed = tp.remove_hashtags("This#notcool is amazing!")
        self.assertEqual(processed, "This is amazing!")

    def test_remove_mentions(self):
        # Test tweet with a single mention
        processed, mentions = tp.remove_mentions("@user1 This is a test.")
        self.assertEqual(processed, " This is a test.")
        self.assertEqual(mentions, ["@user1"])

        # Test tweet with multiple mentions
        processed, mentions = tp.remove_mentions("@user1 @user2 This is a test.")
        self.assertEqual(processed, "  This is a test.")
        self.assertEqual(set(mentions), set(["@user1", "@user2"]))  # Convert to sets for unordered comparison

        # Test tweet with no mentions
        processed, mentions = tp.remove_mentions("This is a test.")
        self.assertEqual(processed, "This is a test.")
        self.assertEqual(mentions, [])

        # Test tweet with repeated mentions
        processed, mentions = tp.remove_mentions("@user1 @user1 This is a test.")
        self.assertEqual(processed, "  This is a test.")
        self.assertEqual(mentions, ["@user1"])  # Mentions should be unique

        # Test tweet where mentions are part of the words
        processed, mentions = tp.remove_mentions("This is a test @work.")
        self.assertEqual(processed, "This is a test .")
        self.assertEqual(mentions, ["@work"])
    
    def test_remove_special_characters(self):
        # Test tweet containing special characters
        processed = tp.remove_special_characters("Th!s @ 1s a t3st.")
        self.assertEqual(processed, "Ths   s a tst")

        # Test tweet containing no special characters
        processed = tp.remove_special_characters("This is a test")
        self.assertEqual(processed, "This is a test")

        # Test tweet containing uppercase letters
        processed = tp.remove_special_characters("This Is A Test")
        self.assertEqual(processed, "This Is A Test")

        # Test tweet containing special characters and numbers
        processed = tp.remove_special_characters("Th!s 1s a t3st!")
        self.assertEqual(processed, "Ths  s a tst")

        # Test tweet containing only special characters
        processed = tp.remove_special_characters("@!#@$%")
        self.assertEqual(processed, "     ")

        # Test empty string
        processed = tp.remove_special_characters("")
        self.assertEqual(processed, "")
    
    def test_remove_extra_spaces(self):
        # Test tweet containing multiple consecutive spaces
        processed = tp.remove_extra_spaces("This  is  a  test.")
        self.assertEqual(processed, "This is a test.")

        # Test tweet containing leading spaces
        processed = tp.remove_extra_spaces("  This is a test.")
        self.assertEqual(processed, "This is a test.")

        # Test tweet containing trailing spaces
        processed = tp.remove_extra_spaces("This is a test.  ")
        self.assertEqual(processed, "This is a test.")

        # Test tweet containing leading and trailing spaces
        processed = tp.remove_extra_spaces("  This is a test.  ")
        self.assertEqual(processed, "This is a test.")

        # Test empty string
        processed = tp.remove_extra_spaces("   ")
        self.assertEqual(processed, "")

        # Test string with only a single space
        processed = tp.remove_extra_spaces(" ")
        self.assertEqual(processed, "")
    
    def test_space_between_emojis(self):
        # Test tweet containing consecutive emojis without spaces
        processed = tp.space_between_emojis("I love this😀😀")
        self.assertEqual(processed, "I love this 😀 😀")

        # Test tweet containing emoji attached to a word
        processed = tp.space_between_emojis("I love this😀")
        self.assertEqual(processed, "I love this 😀")

        # Test tweet containing emojis with existing spaces in between
        processed = tp.space_between_emojis("I love this 😀 😀")
        self.assertEqual(processed, "I love this 😀 😀")

        # Test tweet with emojis separated by extra spaces
        processed = tp.space_between_emojis("I love this 😀  😀")
        self.assertEqual(processed, "I love this 😀 😀")

        # Test tweet with emojis attached to words
        processed = tp.space_between_emojis("Love😍Hate😡")
        self.assertEqual(processed, "Love 😍 Hate 😡")

        # Test empty string
        processed = tp.space_between_emojis("")
        self.assertEqual(processed, "")
    
    def test_preprocess(self):
        # Full pipeline test, with all options enabled
        processed, mentions = tp.preprocess(
            "RT @john: Check out this link! http://example.com #example 😍😍 Sooo cool!!!", 
            delete_emojis=True, 
            extract=True
        )
        self.assertEqual(processed, "check out this link so cool")
        self.assertEqual(mentions, ["@john"])

        # Test without removing emojis and without extracting mentions
        processed = tp.preprocess(
            "RT @john: 😍😍 Sooo cool!!!", 
            delete_emojis=False, 
            extract=False
        )
        self.assertEqual(processed, "😍 😍 so cool")

        # Test with a different exceptions list
        processed, _ = tp.preprocess(
            "RT @john: Checkk out thiss linkk! http://example.com #example 😍😍 Sooo cool!!!", 
            delete_emojis=True, 
            extract=True,
            exceptions=["k", "s"]
        )
        self.assertEqual(processed, "checkk out thiss link so cool")

        # Test with an empty string
        processed, mentions = tp.preprocess(
            "", 
            delete_emojis=True, 
            extract=True
        )
        self.assertEqual(processed, "")
        self.assertEqual(mentions, [])
    
    def test_remove_words(self):
        # Test with a basic example
        processed = tp.remove_words("This is a sample sentence.", ["this", "a", "sample"])
        self.assertEqual(processed, "is sentence.")

        # Test with an empty bag_of_words
        processed = tp.remove_words("This is another sample.", [])
        self.assertEqual(processed, "This is another sample.")

        # Test with a bag_of_words containing special characters
        processed = tp.remove_words("This is .?* another ? sample.", [".", "?", "*", "another"])
        self.assertEqual(processed, "This is sample.")

        # Test with an empty string
        processed = tp.remove_words("", ["this", "is", "empty"])
        self.assertEqual(processed, "")

        # Test with an all-stopword string
        processed = tp.remove_words("This is a sample", ["This", "is", "a", "sample"])
        self.assertEqual(processed, "")
    
    def test_unnest_tokens(self):
        # Test with a simple DataFrame
        df = pd.DataFrame({'text': ['Hello world', 'Python is great']})
        result = tp.unnest_tokens(df, 'text')
        self.assertEqual(result['text'].tolist(), ['Hello', 'world', 'Python', 'is', 'great'])

        # Test with an "id" column
        df = pd.DataFrame({'text': ['Hello world', 'Python is great']})
        result = tp.unnest_tokens(df, 'text', create_id=True)
        self.assertEqual(result['id'].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(result['text'].tolist(), ['Hello', 'world', 'Python', 'is', 'great'])

        # Test with an empty DataFrame
        df = pd.DataFrame({'text': []})
        result = tp.unnest_tokens(df, 'text')
        self.assertTrue(result.empty)

        # Test with a DataFrame that has NaN values
        df = pd.DataFrame({'text': ['Hello world', None]})
        result = tp.unnest_tokens(df, 'text')
        self.assertEqual(result['text'].tolist(), ['Hello', 'world', None])

        # Test with a DataFrame that has a single word in each row
        df = pd.DataFrame({'text': ['Hello', 'Python']})
        result = tp.unnest_tokens(df, 'text')
        self.assertEqual(result['text'].tolist(), ['Hello', 'Python'])

    def test_lemmatization(self):
        # Test a typical word
        result = self.text_preprocessor.spanish_lemmatizer("corriendo", self.nlp)
        self.assertEqual(result, "correr")

        # Test a word with an accent
        result = self.text_preprocessor.spanish_lemmatizer("está", self.nlp)
        self.assertEqual(result, "estar")

        # Test an empty string
        result = self.text_preprocessor.spanish_lemmatizer("", self.nlp)
        self.assertEqual(result, "")

        # Test a non-Spanish word
        result = self.text_preprocessor.spanish_lemmatizer("running", self.nlp)
        self.assertEqual(result, "running")

    def test_error_handling(self):
        # Pass a non-string object and see if it returns the object and logs an error
        result = self.text_preprocessor.spanish_lemmatizer(["not_a_string"], self.nlp)
        self.assertEqual(result, ["not_a_string"])  # Assuming that your function prints an error and returns the original token

    def test_create_bol(self):
        # Assuming preprocess and lemmatize methods are also part of TextPreprocessor
        preprocessed_tweets = [self.text_preprocessor.preprocess(tweet) for tweet in self.sample_tweets]
        lemmatized_tweets = [self.text_preprocessor.spanish_lemmatizer(tweet, self.nlp) for tweet in preprocessed_tweets]

        # Convert the list of lemmatized tweets to a NumPy array
        lemmatized_array = np.array(lemmatized_tweets)

        # Run the create_bol method
        result_df = self.text_preprocessor.create_bol(lemmatized_array)

        # Perform your checks here. For example:
        self.assertIsNotNone(result_df)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue('bow_id' in result_df.columns)
        self.assertTrue('bow_name' in result_df.columns)
        self.assertTrue('lemma' in result_df.columns)
        self.assertTrue('similarity' in result_df.columns)
        self.assertTrue('threshold' in result_df.columns)

    def test_get_most_common_strings(self):
        # Test with flat list of strings
        texts1 = ["hello world", "hello", "world"]
        most_common_strings1 = self.text_preprocessor.get_most_common_strings(texts1, 2)
        self.assertEqual(most_common_strings1, [('hello', 2), ('world', 2)])

        # Test with list of lists
        texts2 = [["hello world"], ["hello"], ["world"]]
        most_common_strings2 = self.text_preprocessor.get_most_common_strings(texts2, 2)
        self.assertEqual(most_common_strings2, [('hello', 2), ('world', 2)])

        # Test with single most common string
        texts3 = ["apple", "apple", "banana"]
        most_common_strings3 = self.text_preprocessor.get_most_common_strings(texts3, 1)
        self.assertEqual(most_common_strings3, [('apple', 2)])

        # Test with empty list
        texts4 = []
        most_common_strings4 = self.text_preprocessor.get_most_common_strings(texts4, 2)
        self.assertEqual(most_common_strings4, [])

        # Test with single-element list
        texts5 = ["single"]
        most_common_strings5 = self.text_preprocessor.get_most_common_strings(texts5, 1)
        self.assertEqual(most_common_strings5, [('single', 1)])
    
    def test_spacy_pipeline_default(self):
        processed_docs = self.text_preprocessor.spacy_pipeline(self.sample_documents)
        # Check that processed_docs is a list of lists
        self.assertIsInstance(processed_docs, list)
        self.assertIsInstance(processed_docs[0], list)
        
    def test_spacy_pipeline_custom_lemmatizer(self):
        nlp = spacy.load("es_core_news_sm")
        custom_lemmatizer = None  # Replace with your custom lemmatizer function
        processed_docs = self.text_preprocessor.spacy_pipeline(self.sample_documents, custom_lemmatizer=custom_lemmatizer)
        # Insert check for custom lemmatization
    
    def test_spacy_pipeline_pipeline_options(self):
        processed_docs = self.text_preprocessor.spacy_pipeline(self.sample_documents, pipeline=['tokenizer'])
        # Insert check for pipeline options, e.g., whether only tokenization was performed
    
    def test_spacy_pipeline_stopwords_language(self):
        spanish_stopwords = stopwords.words('spanish')
        processed_docs = self.text_preprocessor.spacy_pipeline(self.sample_documents, stopwords_language='spanish')
        # Check if stopwords are effectively removed
    
    def test_spacy_pipeline_most_common_strings(self):
        processed_docs, most_common_words = self.text_preprocessor.spacy_pipeline(self.sample_documents, num_strings=2)
    
if __name__ == '__main__':
    unittest.main()