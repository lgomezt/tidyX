import re
import regex
import emoji
from unidecode import unidecode
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Union, Tuple, Optional
from thefuzz import fuzz
import tqdm
import emoji
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import os

class TextPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def remove_repetitions(string: str, exceptions = ["r", "l", "n", "c", "a", "e", "o"]) -> str:
        """
        Removes consecutive repeated characters in a given string, with some optional exceptions.

        For example, the string 'coooroosooo' would be transformed to 'coroso'.

        Args:
            string (str): 
                The text to be processed.
            exceptions (list, optional): 
                A list of characters that can be repeated once consecutively without being removed. Defaults to ['r', 'l', 'n', 'c', 'a', 'e', 'o'].

        Returns:
            str: 
                The processed text with consecutive repetitions removed, except for characters in the exceptions list.
        """

        if not exceptions: # (exceptions == None):
            # Remove all consecutive repetitions
            string = re.sub(r'(.)\1+', r'\1', string)
        else:
            # Allow one repetition for characters in exceptions list
            exceptions_pattern = "|".join(re.escape(char) for char in exceptions)
            string = re.sub(r'({})\1+'.format(exceptions_pattern), r'\1\1', string)
            
            # Remove all other consecutive repetitions
            string = re.sub(r'([^{}])\1+'.format(re.escape(exceptions_pattern)), r'\1', string)

        return string

    @staticmethod
    def remove_last_repetition(string: str, exceptions = ["r", "l", "n", "c", "a", "e", "o"]) -> str:
        """
        Removes the repetition of the last character in each word of a given string.
        
        In Spanish, no word ends with a repeated character. However, in social media,
        it is common to emphasize words by repeating the last character. This function
        cleans the text to remove such repetitions. 
        
        For example, the input "Holaaaa amigooo" would be transformed to "Hola amigo".
        
        Args:
            string (str): 
                The text to be processed.
            exceptions (list, optional): 
                A list of characters that can be repeated once consecutively without being removed. Defaults to ['r', 'l', 'n', 'c', 'a', 'e', 'o'].
            
        Returns:
            str: 
                The processed text with the last character of each word de-duplicated.
        """

        if not exceptions: 
            # Remove all consecutive repetitions
            string = re.sub(r'(\w)\1+\b', r'\1', string) 
        else:
            # Allow one repetition for characters in exceptions list
            exceptions_pattern = "|".join(re.escape(char) for char in exceptions)
            string = re.sub(r'({})\1+\b'.format(exceptions_pattern), r'\1\1', string)
            
            # Remove all other consecutive repetitions
            string = re.sub(r'([^{}])\1+\b'.format(re.escape(exceptions_pattern)), r'\1', string)
        
        return string

    @staticmethod
    def remove_urls(string: str) -> str:
        """
        Removes all URLs that start with "http" from a given string.
        
        This function scans the entire string and removes any sequence of characters that
        starts with "http" and continues until a space or end of line is encountered.
        
        Args:
            string (str): 
                The text to be processed.
            
        Returns:
            str: 
                The processed text with URLs removed.
        """
        
        string = re.sub(r"http\S+", '', string) # re.sub(r"http[^ ]+", '', string)
        string = TextPreprocessor.remove_extra_spaces(string)

        return string 

    @staticmethod
    def remove_RT(string: str) -> str:
        """
        Removes the "RT" prefix from tweets.
        
        This function removes the "RT" prefix that usually appears at the beginning of retweets.
        It accounts for the possibility of varying white-space after "RT".
        
        Args:
            string (str): 
                The tweet text to be processed.
            
        Returns:
            str: 
                The processed tweet text with the "RT" prefix removed if it appears at the beginning.
        """
        
        string = re.sub(r"^RT\s+", '', string) # re.sub("^RT ", '', string)
        string = TextPreprocessor.remove_extra_spaces(string)

        return string

    @staticmethod
    def remove_accents(string: str, delete_emojis = True) -> str:
        """
        Removes accents and optionally emojis from a string.

        This function removes accent marks from characters in a given string. If specified, it can also remove emojis.

        Args:
            string (str): 
                The input string potentially containing accented characters and/or emojis.
            delete_emojis (bool, optional): 
                If True, removes emojis from the string. Default is True.

        Returns:
            str: 
                The string with accented characters and optionally emojis removed.
        """
        
        if delete_emojis:
            string = unidecode(string)
        else:
            string = re.sub(u"[àáâãäå]", 'a', string)
            string = re.sub(u"[èéêë]", 'e', string)
            string = re.sub(u"[ìíîï]", 'i', string)
            string = re.sub(u"[òóôõö]", 'o', string)
            string = re.sub(u"[ùúûü]", 'u', string)
            # Uncomment the next line if you also want to replace "ñ" with "n"
            # string = re.sub(u"[ñ]", 'n', string)
            
        string = TextPreprocessor.remove_extra_spaces(string)

        return string

    @staticmethod
    def remove_hashtags(string: str) -> str:
        """
        Removes hashtags from a given string.
        
        This function scans the string and removes any text that starts with a '#' 
        and is followed by alphanumeric characters.
        
        Args:
            string (str): 
                The text that may contain hashtags.
            
        Returns:
            str: 
                The processed text with hashtags removed.
        """
        
        string = re.sub(r'#\w+', '', string)
        string = TextPreprocessor.remove_extra_spaces(string)

        return string

    @staticmethod
    def remove_mentions(string: str, extract = True):
        """
        Removes mentions (e.g., @username) from a given tweet string.

        This function scans the string and removes any text that starts with '@' 
        followed by the username. Optionally, it can also return a list of unique mentions.

        Args:
            string (str): 
                The tweet text that may contain mentions.
            extract (bool, optional): 
                If True, returns a list of unique mentions. Defaults to True.

        Returns:
            str: 
                The processed tweet text with mentions removed.
            list: 
                If `extract` is True, returns a list of unique mentioned accounts in the tweet.
        """
        
        mentions = []
        # Extract mentions if needed
        if extract:
            mentions = list(set(re.findall(r"@\w+[:]*", string))) # np.unique(re.findall(pattern = "@[^ ]+", string = string))
        
        # Remove mentions
        string = re.sub(r"@\w+[:]*", "", string) # re.sub("@[^ ]+", "", string)
        string = TextPreprocessor.remove_extra_spaces(string)
        
        return string, mentions

    @staticmethod
    def remove_extra_spaces(string: str) -> str:
        """
        Removes extra spaces within and surrounding a given string.
        
        This function trims leading and trailing spaces and replaces any occurrence of consecutive spaces between words with a single space.
        
        Args:
            string (str): 
                The text that may contain extra spaces.
            
        Returns:
            str: 
                The processed text with extra spaces removed.
        """
        
        string = re.sub(" +", " ", string)
        return string.strip()

    @staticmethod
    def remove_special_characters(string: str, allow_numbers: bool = False) -> str:
        """
        Removes all characters from a string except for lowercase letters and spaces.
        
        This function scans the string and removes any character that is not a lowercase letter or a space.
        Optionally, numbers can be retained.
        As a result, punctuation marks, exclamation marks, special characters, and uppercase letters are eliminated.
        
        Args:
            string (str): 
                The text that may contain special characters.
            allow_numbers (bool): 
                Whether to allow numbers in the string. Default is False.
                
        Returns:
            str: 
                The processed text with special characters removed.
        """
    
        pattern = r'[^a-z\p{So} ]+'
        
        if allow_numbers:
            pattern = r'[^a-z0-9\p{So} ]+'

        string = regex.sub(pattern, ' ', string)

        string = TextPreprocessor.remove_extra_spaces(string)
            
        return string

    @staticmethod
    def space_between_emojis(string: str) -> str:
        """
        Inserts spaces around emojis within a string.
        
        This function adds a space before and after each emoji character in the given string to ensure that emojis are separated from other text or emojis. Extra spaces are then removed.

        Args:
            string (str): 
                The text that may contain emojis.

        Returns:
            str: 
                The processed text with spaces inserted around each emoji.
        """
        
        return TextPreprocessor.remove_extra_spaces(''.join((' ' + c + ' ') if c in emoji.EMOJI_DATA else c for c in string))

    @staticmethod
    def preprocess(string: str, delete_emojis: bool = True, extract: bool = False, 
                exceptions: List[str] = ["r", "l", "n", "c", "a", "e", "o"], allow_numbers: bool = False,
                remove_stopwords: bool = False, language_stopwords: str = "spanish", allow_repetition: bool = False) -> Union[str, Tuple[str, List[str]]]:
        """
        Preprocesses a string, typically a tweet, by applying a series of cleaning functions. The function performs the following steps:

        1. Removes the 'RT' prefix from retweeted tweets. 
        2. Converts the entire string to lowercase.
        3. Removes all accents and, if specified, emojis.
        4. Optionally extracts and/or removes all mentions (e.g., @elonmusk).
        5. Removes URLs.
        6. Removes hashtags.
        7. Removes special characters such as !, ?, -, ;, etc. while optionally preserving numbers.
        8. Removes stopwords if indicated.
        9. Removes extra spaces between words.
        10. Reduces consecutive repeated characters, with exceptions defined in the `exceptions` parameter.
        11. Separate consecutive emojis.

        Args:
            string (str): 
                The raw text.
            delete_emojis (bool): 
                If True, removes emojis from the string. Default is True.
            extract (bool): 
                If True, extracts and returns a list of all mentioned accounts in the text. Default is False.
            exceptions (list): 
                Characters that are allowed to be repeated consecutively. Defaults to ['r', 'l', 'n', 'c', 'a', 'e', 'o'].
            allow_numbers (bool): 
                If True, numbers are preserved in the string. Default is False.
            remove_stopwords (bool): 
                If True, stopwords are removed based on the specified language. Default is False.
            language_stopwords (str): 
                The language for which stopwords should be removed. Defaults to "spanish".
            allow_repetition (str):
                If True, will allow repeated characters. Default is False.

        Returns:
            str: 
                The cleaned text.
            mentions (list): 
                If `extract` is True, this list contains all mentioned accounts in the original text.
        """

        string = TextPreprocessor.remove_RT(string)
        string = string.lower()
        string = TextPreprocessor.remove_accents(string, delete_emojis=delete_emojis)
        string, mentions = TextPreprocessor.remove_mentions(string, extract=extract)
        string = TextPreprocessor.remove_urls(string)
        string = TextPreprocessor.remove_hashtags(string)
        string = TextPreprocessor.remove_special_characters(string, allow_numbers=allow_numbers)
        string = TextPreprocessor.remove_words(string, bag_of_words=None, remove_stopwords=remove_stopwords, 
                                            language=language_stopwords)
        string = TextPreprocessor.remove_extra_spaces(string)
        if not allow_repetition:
            string = TextPreprocessor.remove_repetitions(string, exceptions=exceptions)
            string = TextPreprocessor.remove_last_repetition(string, exceptions=exceptions)
        string = TextPreprocessor.space_between_emojis(string)

        return (string, mentions) if extract else string
        
    # Class-level dictionary to cache stopwords for each language
    STOPWORDS_CACHE = {}

    @staticmethod
    def load_stopwords(language='spanish'):
        """
        Load and cache stopwords for a given language.

        Notes:
            To utilize this function, the nltk library must be installed and the stopwords dataset downloaded.
            
            - To install nltk::

                pip install nltk

            - To download the stopwords dataset::

                import nltk
                nltk.download('stopwords')
        """
        
        if language not in TextPreprocessor.STOPWORDS_CACHE:
            # Import stopwords
            stopwords_temp = stopwords.words(language)
            # Remove accents
            stopwords_temp = [unidecode(i) for i in stopwords_temp]
            TextPreprocessor.STOPWORDS_CACHE[language] = stopwords_temp


    @staticmethod
    def remove_words(string: str, bag_of_words: list = None, remove_stopwords: bool = False, language: str = 'spanish') -> str:
        """Removes specified words and optionally stopwords from a string.

        Args:
            string (str): 
                The input string from which words are to be removed.
            bag_of_words (list, optional): 
                A list of words that should be removed from the string. Defaults to None.
            remove_stopwords (bool, optional): 
                If True, removes predefined stopwords from the string based on the specified language. Defaults to False.
            language (str, optional): 
                Language of the stopwords that will be removed if `remove_stopwords` is set to True. Defaults to 'spanish'.

        Returns:
            str: 
                A string with the specified words removed.

        Notes:
            To utilize this function, the nltk library must be installed and the stopwords dataset downloaded.
            
            - To install nltk::

                pip install nltk

            - To download the stopwords dataset::

                import nltk
                nltk.download('stopwords')
        """
        
        # If remove_stopwords is True, get the cached stopwords for the specified language
        if remove_stopwords:
            TextPreprocessor.load_stopwords(language)
            if not bag_of_words:
                bag_of_words = []
            bag_of_words.extend(TextPreprocessor.STOPWORDS_CACHE[language])
        
        # Return original string if no bag_of_words is provided
        if not bag_of_words:
            return string
        
        # Create a regex pattern to match any word from bag_of_words surrounded by word boundaries
        pattern = r'\b(?:{})\b'.format('|'.join(re.escape(word) for word in bag_of_words))
        
        # Remove words from the string that match the pattern
        string = re.sub(pattern, '', string)
        
        # Remove extra spaces
        string = TextPreprocessor.remove_extra_spaces(string)

        return string

    @staticmethod
    def unnest_tokens(df: pd.DataFrame, input_column: str, id_col: str = None, unique: bool = False) -> pd.DataFrame:
        """
        Unnests or flattens a DataFrame by tokenizing a specified column.

        Given a pandas DataFrame and a column name, this function splits the specified column on spaces,
        turning each token into a separate row in the resulting DataFrame.

        Args:
            df (pd.DataFrame): 
                The input DataFrame. Each row is expected to represent a document.
            input_column (str): 
                The name of the column to tokenize.
            id_col (str, optional): 
                The name of the column that uniquely identifies each document. If None, 
                an "id" column is added based on the DataFrame's index. Defaults to None.
            unique (bool, optional):
                If True, it will deduplicate tokens and concatenate the IDs where they appear. 
                If False, every token will have a corresponding row. Defaults to False.

        Returns:
            pd.DataFrame: 
                A DataFrame where each row corresponds to a token from the input column.
        """
        
        # Check if input_column exists in the dataframe
        if input_column not in df.columns:
            raise ValueError(f"'{input_column}' is not a column in the provided DataFrame.")
        
        # If id_col is None, create an "id" column based on the DataFrame's index
        if not id_col:
            df["id"] = df.index
            id_col = "id"
        
        # Tokenize the specified column
        df[input_column] = df[input_column].astype(str).str.split(" ")

        # Explode the DataFrame to create one row per token
        df = df.explode(input_column)

        # If unique is True, deduplicate tokens and concatenate the IDs where they appear
        if unique:
            df = df.groupby(input_column)[id_col].agg(lambda x: ", ".join(map(str, x))).reset_index()

        return df
        
    @staticmethod
    def create_bol(lemmas: np.ndarray, verbose: bool = False) -> pd.DataFrame:
        """
        Groups lemmas based on Levenshtein distance to handle misspelled words in social media data.
        
        This method clusters lemmas that are similar to each other based on their 
        Levenshtein distance. The aim is to group together possibly misspelled 
        versions of the same lemma.
        
        Args: 
            lemmas (np.ndarray): 
                An array containing lemmas to be grouped.
            verbose (bool, optional): 
                If set to True, progress will be printed at every 5% increment. Defaults to False.
        
        Returns: 
            pd.DataFrame: A DataFrame with columns: 
                - "bow_id": An ID for the bag of lemmas (int).
                - "bow_name": The representative name for the bag of lemmas (str).
                - "lemma": The original lemma (str).
                - "similarity": The similarity score based on fuzz.ratio (int).
                - "threshold": The similarity threshold used (int).
                
        Notes:
            The method utilizes the `fuzz.ratio` function to determine similarity 
            between lemmas. The threshold for similarity depends on the length of 
            the lemma being compared, to accommodate the sensitivity of `fuzz.ratio` 
            towards shorter words.
        """
        # Create an empty dataframe to store the bags of words (lemmas)
        bow_df = pd.DataFrame()
        # How many lemmas do we have?
        num_lemmas = len(lemmas)
        # Iterator
        iterator = 0
        # Step to show the progress every 5%
        step = max(1, int(num_lemmas * 0.05)) 

        # lemmas is an array that will reduce its size because when an element of lemmas is assigned to a bag of
        # words, it will be dropped from the array. When all the lemmas have been assigned to a bag of words,
        # lemmas array will be empty.
        while len(lemmas) > 0:
            try:
                # Pick a lemma: lemma i
                current_lemma = lemmas[0]
                # Let calculate the distance between the lemma i and the other lemmas
                distances = np.array([fuzz.ratio(current_lemma, lemma) for lemma in lemmas])
                # fuzz.ratio is very sensible to small words so is important to control for this
                threshold = 88 if len(current_lemma) < 3 else 87 if len(current_lemma) <= 4 else 86 if len(current_lemma) == 5 else 85
                
                # Find the position inside the array of the lemmas that have threshold% coincidence with lemma i
                similar_lemmas_idx = np.where(distances > threshold)[0]
                # Create bag_i
                similar_lemmas = lemmas[similar_lemmas_idx]

                # Compile the information in a dataframe
                bag_data = pd.DataFrame({
                    "bow_id": iterator + 1,
                    "bow_name": similar_lemmas[0],
                    "lemma": similar_lemmas,
                    "similarity": distances[similar_lemmas_idx],
                    "threshold": threshold
                })

                # Delete the words that were already assigned to a bag of words.
                # The reason is because the lemma i is near to the lemma j, the opposite is also true.
                # And if exists another word k that is near to j, we also expect that the word k also be near to the word i
                lemmas = np.delete(lemmas, similar_lemmas_idx)
                # Put the results in the important DataFrame
                bow_df = pd.concat([bow_df, bag_data])
            
                # Progress indicator
                if verbose and iterator % step == 0:
                        progress = np.round(100 - (len(lemmas) / num_lemmas * 100), 2)
                        print(f"{progress}% completed.")
                        
                iterator += 1
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        
        return bow_df.reset_index(drop = True)
    
    @staticmethod  
    def get_most_common_strings(texts: Union[str, List[str], pd.Series], num_strings: Union[int, None] = None, as_dataframe: bool = False) -> Union[List[Tuple[str, int]], pd.DataFrame]:
        """
        Retrieves the most common strings in a list of texts. 
        
        This method serves primarily to validate preprocessing steps or to
        provide descriptive information about a collection of texts. It can handle
        both flat lists of strings, lists of lists of strings, and pandas Series.
        
        Args:
            texts (Union[str, List[str], pd.Series]): 
                Texts can be a single string, a list of strings, or a pandas Series.
            num_strings (Union[int, None], optional): 
                The number of most common strings to be returned. If None, all unique strings will be returned. Defaults to None.
            as_dataframe (bool, optional):
                If True, the result will be returned as a DataFrame. Otherwise, it will be a list of tuples. Defaults to False.
        
        Returns:
            Union[List[Tuple[str, int]], pd.DataFrame]: 
                Depending on the `as_dataframe` parameter, either a list of tuples or a DataFrame will be returned,
                where each entry contains a string and its occurrence count.
        
        Example:
            >>> TextPreprocessor.get_most_common_strings(["apple orange", "apple banana"], 1)
            [('apple', 2)]
            
            >>> TextPreprocessor.get_most_common_strings([["apple", "orange"], ["apple", "banana"]], 1)
            [('apple', 2)]
            
        Raises:
            ValueError: If `texts` is an empty list or if `num_strings` is non-positive.
        """
        
        if isinstance(texts, pd.Series):
            texts = texts.values.tolist()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("texts must be a non-empty list, string, or pandas Series.")
        
        if num_strings is not None and num_strings <= 0:
            raise ValueError("num_strings must be a positive integer or None.")
        
        word_counts = defaultdict(int)

        # If texts contain sublists, flatten them.
        if any(isinstance(i, list) for i in texts):
            texts = [item for sublist in texts for item in sublist]
        
        # Update word counts after splitting each text into words.
        for text in texts:
            for word in str(text).split():
                word_counts[word] += 1

        # Get the most common words based on the count specified
        results = Counter(word_counts).most_common(num_strings)
        
        # Return results as DataFrame if requested
        if as_dataframe:
            return pd.DataFrame(results, columns=["String", "Count"])
        return results
    
    def load_data(file: str = "spanish") -> pd.DataFrame:
        """
        Load a dataset based on the specified language or file name.
        
        This function is primarily used to load a dataset that contains tweets. The default dataset,
        "Tweets sobre migrantes venezolanos.xlsx", consists of 1,000 tweets from Colombia about Venezuelan migrants.
        
        Args:
            file (str, optional): 
                Specifies the name of the file or the language of the dataset. If "spanish" is provided,
                the function will default to loading the dataset "Tweets sobre migrantes venezolanos.xlsx".
                Default is "spanish".
        
        Returns:
            pd.DataFrame: 
                A pandas DataFrame containing the loaded dataset.
        
        Raises:
            FileNotFoundError: 
                If the specified file is not found in the 'data' directory.
        
        Note:
            Ensure you have the 'openpyxl' library installed as it is required to read .xlsx files.
        
        Example:
            >>> load_data("spanish")
            ... # Returns a DataFrame with 1,000 tweets about Venezuelan migrants from Colombia.
        """
        
        if file == "spanish":
            file = "Tweets sobre migrantes venezolanos.xlsx"
            
        # Get the current directory of the module and append the relative path to the xlsx file
        file_path = os.path.join(os.path.dirname(__file__), 'data', file)

        # Read the xlsx file into a DataFrame
        df = pd.read_excel(file_path, engine='openpyxl')

        return df





