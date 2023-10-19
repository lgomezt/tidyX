import tqdm
import spacy
from typing import List, Union, Tuple
from spacy.lang.es import Spanish
from spacy.language import Language
from .text_preprocessor import TextPreprocessor 
import warnings
import spacy_spanish_lemmatizer

class SpacyPreprocessor:

    def __init__(self):
        pass

    @staticmethod
    def spanish_lemmatizer(token: str, model: Spanish) -> str:
        """Lemmatizes a given token using Spacy's Spanish language model.
        
        Note: Before using this function, a Spacy model for Spanish should be downloaded.
        Use `python -m spacy download name_of_model` to download a model. Available models:
        "es_core_news_sm", "es_core_news_md", "es_core_news_lg", "es_dep_news_trf".
        For more information, visit https://spacy.io/models/es

        Args:
            token (str): The token to be lemmatized.
            model (spacy.lang.es.Spanish): A Spacy language model object.

        Returns:
            str: The lemmatized version of the token, with accents removed.
        """
        import spacy

        if not token:
            return token
        
        try:
            lemma = model(token)[0].lemma_
            lemma = TextPreprocessor.remove_accents(lemma)
            return lemma
        except Exception as e:
            print(f"An error occurred: {e}")
            return token

    def is_component_registered(name: str) -> bool:
        """
        Check if a spaCy pipeline component with the given name is already registered.
        
        Args:
            name (str): 
                The name of the spaCy pipeline component.
                
        Returns:
            conditional (bool) True if the component is already registered. False otherwise.
        """
        return name in Language.factories

    def register_component():
        """
        Conditionally register the custom_lemmatizer component.
        """
        if not SpacyPreprocessor.is_component_registered("custom_lemmatizer"):

            @Language.factory("custom_lemmatizer")
            def custom_lemmatizer(nlp: Spanish, name: str) -> 'spacy_spanish_lemmatizer.main.create_spanish_lemmatizer':
                """
                Creates and returns a Spanish rule-based lemmatizer for spaCy.
                
                This factory function leverages the `spacy-spanish-lemmatizer` package
                to provide a rule-based lemmatizer for the Spanish language, enhancing
                the spaCy pipeline's capability to extract the base or dictionary form
                of a word, which is crucial for various NLP tasks like text normalization,
                text analysis, and information retrieval.
                
                For additional details on the lemmatizer, refer to:
                https://github.com/pablodms/spacy-spanish-lemmatizer
                
                Args:
                    nlp (spacy.lang.es.Spanish): 
                        The spaCy language model object.
                    name (str): 
                        The name of the lemmatizer, utilized by spaCy to register the component.
                    
                Returns:
                    spacy_spanish_lemmatizer.main.create_spanish_lemmatizer: 
                        A Spanish rule-based lemmatizer for spaCy.
                        
                Example:
                    >>> import spacy
                    >>> nlp = spacy.load('es_core_news_sm')
                    >>> nlp.add_pipe('custom_lemmatizer', name='lemmatizer')
                    >>> doc = nlp("El gato está en la casa")
                    >>> [token.lemma_ for token in doc]
                    ['El', 'gato', 'estar', 'en', 'el', 'casa']
                """
                return spacy_spanish_lemmatizer.main.create_spanish_lemmatizer(nlp, name)

    @staticmethod
    def spacy_pipeline(documents: List[str], custom_lemmatizer: bool = False, pipeline: List[str] = ['tokenize', 'lemmatizer'],
                       stopwords_language: str = 'Spanish', model: str = 'es_core_news_sm', num_strings: int = 0) -> Union[List[List[str]], Tuple[List[List[str]], List[Tuple[str, int]]]]:
        """
        Processes documents through the spaCy pipeline, performing lemmatization and stopword removal, and optionally utilizing a custom lemmatizer for Spanish.
        
        For further information on the custom lemmatizer, refer to:
        https://github.com/pablodms/spacy-spanish-lemmatizer
        
        Note: 
        Ensure the relevant spaCy model is downloaded using:
        ```sh
        python -m spacy download <model_name>
        ```
        where <model_name> can be "es_core_news_sm", "es_core_news_md", "es_core_news_lg", or "es_dep_news_trf".
        
        Args:
            documents (List[str]): A list of texts to be processed.
            custom_lemmatizer (bool, optional): If True, a custom Spanish rule-based lemmatizer is added to the pipeline.
            pipeline (List[str], optional): A list of spaCy pipeline components for processing the documents. Defaults to ['tokenize', 'lemmatizer'].
            stopwords_language (str, optional): Language for the nltk stopwords list. Defaults to 'Spanish'.
            model (str, optional): spaCy model to be used. Defaults to 'es_core_news_sm'.
            num_strings (int, optional): Number of most common strings to return. If 0, only processed documents are returned. Defaults to 0.
        
        Returns:
            Union[List[List[str]], Tuple[List[List[str]], List[Tuple[str, int]]]]:
            A list of processed documents and, if num_strings > 0, a list of the most common strings in the documents.
        
        Raises:
            ValueError: If the documents list is empty.
        """
        import spacy
        import nltk
        from nltk.corpus import stopwords

        if not documents:
            raise ValueError("The documents list must not be empty.")
        try:
            nlp = spacy.load(model, disable=[comp for comp in spacy.lang.es.LANGUAGES_DICT['es'].pipe_names if comp not in pipeline])
        except:
            default_pipeline = ['tagger', 'parser', 'ner', 'lemmatizer']  # Adjust this list based on the components of the specific model you're using
            nlp = spacy.load(model, disable=[comp for comp in default_pipeline if comp not in pipeline])

        # Download resources
        nltk.download('stopwords')
        spanish_stopwords = stopwords.words(stopwords_language)

        if custom_lemmatizer:
            custom_lemmatizer_name = "custom_lemmatizer"
            if 'lemmatizer' in nlp.pipe_names:
                nlp.replace_pipe("lemmatizer", custom_lemmatizer_name)
            else:
                nlp.add_pipe(custom_lemmatizer_name, name=custom_lemmatizer_name, last=True)

        processed_documents = [
            [TextPreprocessor.remove_accents(token.lemma_) for token in nlp(doc) if token.text not in spanish_stopwords and token.lemma_ not in spanish_stopwords]
            for doc in tqdm.tqdm(documents, total=len(documents))
        ]

        if num_strings > 0:
            most_common_words = TextPreprocessor.get_most_common_strings(processed_documents, num_strings)
            return processed_documents, most_common_words
        
        return processed_documents

SpacyPreprocessor.register_component()


