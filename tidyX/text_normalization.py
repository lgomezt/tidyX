import spacy
from typing import List, Union, Tuple
from .text_preprocessor import TextPreprocessor
from nltk.stem.snowball import SnowballStemmer

class TextNormalization:

    def __init__(self):
        pass

    @staticmethod
    def spanish_lemmatizer(token: str, model: str = 'es_core_news_sm') -> str:
        """Lemmatizes a given token using Spacy's Spanish language model.

        Lemmatization is the process of reducing a word to its base or dictionary form. 
        For example, the word "running" would be lemmatized to "run". Lemmatization takes 
        into account the meaning of the word in the sentence, leveraging vocabulary and 
        morphological analysis.

        Note: Before using this function, a Spacy model for Spanish should be downloaded.
        Use `python -m spacy download name_of_model` to download a model. Available models:
        "es_core_news_sm", "es_core_news_md", "es_core_news_lg", "es_dep_news_trf".
        For more information, visit https://spacy.io/models/es

        Args:
            token (str): The token to be lemmatized.
            model (str, optional): The spaCy language model to use. Defaults to 'es_core_news_sm'.

        Returns:
            str: The lemmatized version of the token, with accents removed.
        """
        if not token:
            return token
        
        try:
            nlp = spacy.load(model)
            lemma = nlp(token)[0].lemma_
            lemma = TextPreprocessor.remove_accents(lemma)
            return lemma
        except Exception as e:
            print(f"An error occurred: {e}")
            return token
        
    def spanish_stemmer(token: str) -> str:
        """Stems a given token using Snowball stemmer for Spanish.

        Stemming is the process of reducing a word to its word stem, often by stripping 
        suffixes. For instance, the word "running" might be stemmed to "run". Unlike 
        lemmatization, stemming doesn't always produce a valid word and doesn't consider 
        the meaning of a word in the context.

        Note: Before using this function, you might need to install nltk if not done already.
        Use `pip install nltk`.

        Args:
            token (str): The token to be stemmed.

        Returns:
            str: The stemmed version of the token.
        """
        if not token:
            return token

        stemmer = SnowballStemmer("spanish")
        
        try:
            stemmed = stemmer.stem(token)
            return stemmed
        except Exception as e:
            print(f"An error occurred: {e}")
            return token