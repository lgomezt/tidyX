import re
from unidecode import unidecode
import numpy as np
from thefuzz import fuzz
import pandas as pd
import regex
import emoji

def remove_repetitions(string, exceptions = ["r", "l", "n", "c", "a", "e", "o"]):
    """This function deletes any consecutive repetition of characters in a string. For example, the string 'coooroosooo' will be changed to 'coroso'. As in many languages it's common to have some special characters that can be repeated, for example the 'l' in spanish to form 'll', the exception argument could be used to specify which characters are allowed to repeat once.

    Args:
        string (str): Text to be formatted.
        exceptions (list): List of characters that are allowed to have two repetitions. For example, if exceptions is ['r'] words like 'carrro' are going to be cleaned as 'carro'.  Defaults to ['r', 'l', 'n', 'c', 'a', 'e', 'o'].

    Returns:
        str: Initial text without consecutive repetition of characters (exceptions may apply, see Args).
    """
    if (exceptions == None):
        string = re.sub('(.)\\1+', r'\1', string)
    else:
        # Only allow one repetition of the exceptions
        exceptions = "|".join(exceptions)
        string = re.sub('({})\\1+'.format(exceptions), r'\1\1', string)
        # Delete the repetition of the other words
        string = re.sub('([^{}])\\1+'.format(exceptions), r'\1', string)
    return string

def remove_last_repetition(string): 
    """As in spanish there is no word that ends with the repetition of any character, but it's frequent in
    social media that people repeat the last character to add more emotion in their messages, this functions
    cleans a string to remove all repetitions of the last character of each word in a sentence. For example, if
    the input of this function is "Holaaaa amigooo" it will be transformed to "Hola amigo".

    Args:
        string (str): Text to be formatted.

    Returns:
        str: Transformed string
    """
    return re.sub(r'(\w)\1+\b', r'\1', string)

def remove_urls(string):
    """Remove all sentences that begins with http

    Args:
        string (str): Text to be formatted.

    Returns:
        str: String without links that begins with html
    """
    return re.sub(r"http[^ ]+", '', string)

def remove_RT(string):
    """Remove RT at the beginning of the tweets

    Args:
        string (str): Text to be formatted.

    Returns:
        str: String without "RT" at the beginning of it.
    """
    return re.sub("^RT ", '', string)

def remove_accents(string, delete_emojis = True):
    """Remove accents and emojis of a string

    Args:
        string (str): String with accents.
        delete_emojis (bool): Decide whether it is necessary to remove the emojis from the string. The default value
            is set to True.
    Returns:
        str: String without accents.
    """

    if delete_emojis:
        string = unidecode(string)
    else:
        string = re.sub(u"[àáâãäå]", 'a', string)
        string = re.sub(u"[èéêë]", 'e', string)
        string = re.sub(u"[ìíîï]", 'i', string)
        string = re.sub(u"[òóôõö]", 'o', string)
        string = re.sub(u"[ùúûü]", 'u', string)
        # string = re.sub(u"[ñ]", 'n', string)
    return string

def remove_hashtags(string):
    """Remove hashtags

    Args:
        string (str): string with hashtag.

    Returns:
        str: string without hashtag.
    """
    return re.sub('#\\w+', '', string)

def remove_mentions(string, extract = True):
    """Remove mentions in a tweet, i.e. @name_of_user.

    Args:
        string (str): String with mentions.
        extract (bool, optional): If it's True, the function will return a list with all accounts mentioned in the string. Defaults to True.

    Returns:
        string (str): String without mentions.
        mentions (list): if extract = True, then mentions will be a list with the mentioned accounts in the tweet.
    """

    # Extract mentions
    if extract:
        mentions = np.unique(re.findall(pattern = "@[^ ]+", string = string))

    # Remove mentions
    string = re.sub("@[^ ]+", "", string)

    if extract:
        return string, mentions
    else:
        return string

def remove_special_characters(string):
    """Removes all characters other than lowercase letters. Therefore, punctuation marks, 
    exclamation marks, special characters, capital letters, etc. are eliminated.

    Args:
        string (str): String with special characters.

    Returns:
        str: String without special characters.
    """
    # return re.sub('[^a-z ]+', '', string)
    return regex.sub('[^a-z\p{So} ]+', '', string)

def remove_extra_spaces(string):
    """Remove all extra spaces between words, at the beginning or at the end of any sentence.

    Args:
        string (str): String with extra spaces.

    Returns:
        str: String without extra spaces.
    """
    string = re.sub(" +", " ", string)
    return string.strip()

def space_between_emojis(string):
    """_summary_

    Args:
        string (_type_): _description_

    Returns:
        _type_: _description_
    """
    return remove_extra_spaces(''.join((' '+c+' ') if c in emoji.UNICODE_EMOJI['en'] else c for c in string))

def preprocess(string, delete_emojis = True, extract = True, 
               exceptions = ["r", "l", "n", "c", "a", "e", "o"]):
    """This function compile other cleaning functions of tidytweets to ease de cleaning process of 
    tweets. The steps that this function makes are:

    1. Remove the 'RT' string at the beginning of the retweeted tweets. (remove_RT)

    2. Lowercase all the string. (.lower)

    3. Remove all the accents and emojis. We suggest to treat the emojis separately to this cleaning process. (remove_accents)

    4. Save and/or remove all the mentions i.e. @elonmusk. (remove_mentions)

    5. Remove the urls. (remove_urls)

    6. Remove hashtags. (remove_hashtags)

    7. Remove special characters i.e. !?-;, among others. (remove_special_characters)

    8. Remove extra spaces. (remove_extra_spaces)
    
    9. Remove repetitions of characters, with some exceptions defined in the `exceptions` parameter. (remove_repetitions and remove_last_repetition)

    Args:
        string (str): Raw tweet
        delete_emojis (bool): Decide whether it is necessary to remove the emojis from the string. The default value
            is set to True.
        extract (bool): If it's True, the function will return a list with 
            all accounts mentioned in the string. Defaults to True.
        exceptions (list): List of characters that are allowed to have two repetitions. For example,
            if exceptions is ['r'] words like 'carrro' are going to be cleaned as 'carro'. 
            Defaults to ['r', 'l', 'n', 'c', 'a', 'e', 'o'].

    Returns:
        str: Tweet cleaned.
    """

    # Remove RT at the beginning of the tweets
    string = remove_RT(string)
    # Lowercase all characters
    string = string.lower()
    # Remove accents:
    string = remove_accents(string, delete_emojis = delete_emojis)
    # Extract and remove all mentions
    if extract:
        string, mentions = remove_mentions(string, extract = True)
    else:
        string = remove_mentions(string, extract = False)
    # Remove links
    string = remove_urls(string)
    # Remove hashtags
    string = remove_hashtags(string)
    # Remove special characters:
    string = remove_special_characters(string)
    # Allow only one space between words
    string = remove_extra_spaces(string)
    # Remove repetited characters
    string = remove_repetitions(string, exceptions = exceptions)
    # Remove repetitions in the last charcater
    string = remove_last_repetition(string)

    if extract:
        return string, mentions
    else:
        return string
    
def remove_words(string, bag_of_words):
    """This function delete all the words listed in bag_of_words from the string. It specially
    useful to remove stopwords. Be careful with the words that are listed in bag_of_words because
    this function will search for exact match. Therefore, the function will not delete words from 
    the string if they are not written in the same format as in the bag_of_words.

    Args:
        string (str): String with unwanted words.
        bag_of_words (list): List of unwanted words.

    Returns:
        str: String without unwanted words.
    """
    # Create a regex pattern to identify any word in bag_of_words
    pat = r'\b(?:{})\b'.format('|'.join(bag_of_words))
    string = re.sub(pat, '', string)
    # Allow only one space between words and remove any leading and trailing space
    string = remove_extra_spaces(string)

    return string

def unnest_tokens(df, input_column, create_id = True):
    """Split a column into tokens, flattening the table into one-token-per-row. The column to split is
    the "input_column" column and the tokens will be separated by simple spaces (" ").

    Args:
        df (DataFrame): A pandas DataFrame
        input_column (str): Name of input column that gets split.
        create_id (bool): If is True, the index of the DataFrame will become the id column. Defaults to True.

    Returns:
        DataFrame: A dataframe where each row is a token.
    """
    if create_id:
        df = df.reset_index().rename(columns = {"index": "id"})

    df[input_column] = df[input_column].str.split(" ")
    df = df.explode(input_column)
    return df

def spanish_lemmatizer(token, model):
    """Use Spacy's pipeline to lemmatize a token. 
    
    Before using this function, you must have downloaded one of the models. With the following code, you
    can do it directly from the terminal: python -m spacy download name_of_model (where name_of_model could
    be "es_core_news_sm", "es_core_news_md", "es_core_news_lg", "es_dep_news_trf"). To see more information, 
    visit: https://spacy.io/models/es

    Args:
        token (str): A token to be lemmatized.
        model (spacy.lang.es.Spanish): Spacy's model loaded using spacy.load("name_of_model").

    Returns:
        str: Lemmatized token
    """
    if token == "":
        return token
    else:
        lemma = model(token)[0].lemma_
        # Lemmatization could create tokens with accents
        lemma = remove_accents(lemma)
        return lemma
    
def create_bol(lemmas, verbose = True):
    """In social media is common that the people make orthographic mistakes, therefore, when we count tokens, 
    misspelled words are not going to sum correctly. This function groups words based on Levenshtein distance.

    Args:
        lemmas (np.array): Numpy array of lemmas.

    Returns:
        DataFrame: With 4 columns ["bow_id", "bow_name", "lemma", "threshold"]. Each row represents a word of the input array (lemmas) and the bag of words/lemmas that it belongs.
    """
    
    # Create an empty dataframe to store the bags of words (lemmas)
    bow = pd.DataFrame()
    # How many lemmas do we have?
    n = len(lemmas)
    # Iterator
    iterator = 0
    # Step to show the progress every 5%
    step = n*0.05//1

    # lemmas is an array that will reduce its size because when an element of lemmas is assigned to a bag of
    # words, it will be dropped from the array. When all the lemmas have been assigned to a bag of words,
    # lemmas array will be empty.
    while len(lemmas) != 0:
        # Pick a lemma: lemma i
        l = lemmas[0]
        # Let calculate the distance between the lemma i and the other lemmas
        d = np.array([fuzz.ratio(l, i) for i in lemmas])
        # fuzz.ratio is very sensible to small words so is important to control for this
        threshold = np.where(len(l) == 3, 87, 
                        np.where(len(l) == 4, 87,
                                np.where(len(l) == 5, 86, 
                                         np.where(len(l) >= 6, 85, 88))))
        # Find the position inside the array of the lemmas that have threshold% coincidence with lemma i
        idx = np.where(d > threshold)[0]
        # Create bag_i
        bag_i = lemmas[idx]
        bag_name_i = bag_i[0]
        # Compile the information in a dataframe
        bag_i_frame = pd.DataFrame([(iterator + 1, bag_name_i, i, j, threshold) for i,j in zip(bag_i, d[idx])],
                                   columns = ["bow_id", "bow_name", "lemma", "similarity", "threshold"])
        # Delete the words that were already assigned to a bag of words.
        # The reason is because the lemma i is near to the lemma j, the opposite is also true.
        # And if exists another word k that is near to j, we also expect that the word k also be near to the word i
        lemmas = np.delete(lemmas, idx)
        # Put the results in the important DataFrame
        bow = pd.concat([bow, bag_i_frame], axis = 0)
        
        # Progress indicator
        if verbose:
            if iterator % step == 0:
                progreso = np.round(100 - len(lemmas)/n * 100, 2)
                print(str(progreso) + "%")
        iterator += 1
    
    return bow.reset_index(drop = True)