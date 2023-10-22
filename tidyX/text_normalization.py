import spacy
from typing import List, Union, Tuple
from spacy.language import Language
from .text_preprocessor import TextPreprocessor
from nltk.stem.snowball import SnowballStemmer
import emoji

class TextNormalization:

    def __init__(self):
        pass

    @staticmethod
    def is_emoji(s: str) -> bool:
        """Check if a given string is an emoji."""
        return s in emoji.EMOJI_DATA

    @staticmethod
    def lemmatizer(token: str, model: Language) -> str:
        """Lemmatizes a given token using Spacy's Spanish language model.

        Lemmatization is the process of reducing a word to its base or dictionary form. 
        For example, the word "running" would be lemmatized to "run". Lemmatization takes 
        into account the meaning of the word in the sentence, leveraging vocabulary and 
        morphological analysis.

        Note: Before using this function, a Spacy model should be downloaded.
        Use `python -m spacy download name_of_model` to download a model. Available models for Spanish are:
        "es_core_news_sm", "es_core_news_md", "es_core_news_lg", "es_dep_news_trf".
        For more information, visit https://spacy.io/models/

        Args:
            token (str): The token to be lemmatized.
            model (spacy.language.Language): A Spacy language model object.

        Returns:
            str: The lemmatized version of the token, with accents removed.
        """
        
        if not token or TextNormalization.is_emoji(token):
            return token
        
        try:
            lemma = model(token)[0].lemma_
            lemma = TextPreprocessor.remove_accents(lemma)
            return lemma
        except Exception as e:
            print(f"An error occurred: {e}")
            return token
        
    def stemmer(token: str, language: str = "spanish") -> str:
        """Stems a given token using Snowball stemmer.

        Stemming is the process of reducing a word to its base or root form, often by stripping 
        suffixes. For instance, the word "running" might be stemmed to "run". Unlike 
        lemmatization, stemming doesn't always produce a valid word and doesn't consider 
        the meaning of a word in the context.

        This function uses the Snowball stemmer, which supports multiple languages including Spanish.

        Note: Before using this function, you might need to install nltk if not done already.
        Use `pip install nltk`.

        Args:
            token (str): The token to be stemmed.
            language (str, optional): The language of the token. Defaults to "spanish".

        Returns:
            str: The stemmed version of the token.
        """
        
        if not token or TextNormalization.is_emoji(token):
            return token

        stemmer = SnowballStemmer(language)
        
        try:
            stemmed = stemmer.stem(token)
            return stemmed
        except Exception as e:
            print(f"An error occurred: {e}")
            return token
