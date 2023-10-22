# tidyX

`tidyX` is a Python package designed for cleaning and preprocessing text for machine learning applications, **especially for text written in Spanish and originating from social networks.** This library provides a complete pipeline to remove unwanted characters, normalize text, group similar terms, etc. to facilitate NLP applications.

## Installation

Install the package using pip:

```bash
pip install tidyX
```

Make sure you have the necessary dependencies installed. If you plan on lemmatizing, you'll need `spaCy` along with the appropriate language models. For Spanish lemmatization, we recommend downloading the `es_core_web_sm` model:

```bash
python -m spacy download es_core_web_sm 
```

For English lemmatization, we suggest the `en_core_web_sm` model:

```bash
python -m spacy download en_core_web_sm 
```

To see a full list of available models for different languages, visit `Spacy's documentation <https://spacy.io/models/>`_.


## Features

- **Text Cleaning**: Strips tweets of unnecessary clutter such as special characters, emojis, and URLs to make the text more digestible for ML models.
- **Emoji Handling**: Provides tools for working with emojis, allowing for their removal or conversion into textual descriptions.
- **Language Specific Preprocessing**: Tailored preprocessing functionalities that consider the linguistic peculiarities of tweets, enhancing the quality of the cleaned text.
- **Dependency Parsing Visualization**: Incorporates visualization tools that enable the display of dependency parses, facilitating linguistic analysis and feature engineering.


## Usage

Here are some basic examples demonstrating how to use `tidyX`:

### Text Preprocessing

The `preprocess` method in the `TextPreprocessor` class is a powerful tool that performs a comprehensive cleaning and preprocessing of tweet texts to prepare them for further analysis or machine learning tasks. It is designed to be highly configurable, allowing you to choose which preprocessing steps to apply.

Here’s how to use the `preprocess` method:

```python
from tidyX import TextPreprocessor

# Creating a TextPreprocessor object
text_preprocessor = TextPreprocessor()

# Raw tweet example
raw_tweet = "RT @user: Check out this link: https://example.com 🌍 #example 😃"

# Applying the preprocess method
cleaned_text, mentions = text_preprocessor.preprocess(raw_tweet)

# Printing the cleaned text and extracted mentions
print("Cleaned Text:", cleaned_text)
print("Mentions:", mentions)
```

In this example, the `preprocess` method is applied to a raw tweet. The method:
- Removes the 'RT' prefix from retweeted tweets
- Converts the text to lowercase
- Removes accents and emojis (configurable)
- Extracts and removes mentions, returning them as a list if required
- Removes URLs, hashtags, and special characters
- Eliminates extra spaces and consecutive repeated characters (with configurable exceptions)

This versatile method allows for a tailored preprocessing approach, adapting to the specific needs of your text analysis or NLP tasks.

### Bag of Lemmas (BoL) Creation

The `create_bol` function is designed to group lemmas based on the Levenshtein distance. This is particularly helpful to manage misspelled words frequently encountered in social media data.

#### **Functionality**

The function creates "bags" that group similar lemmas together by calculating the Levenshtein distance between words, allowing for the grouping of words that are likely variations or misspellings of each other.

##### **Usage**

```python
import numpy as np
import pandas as pd
from tidyX import create_bol

# Example lemmas
lemmas = np.array(["running", "runing", "jogging", "joging"])

# Creating bags of lemmas
bol_df = create_bol(lemmas)

print(bol_df)
```

##### **Parameters**

- `lemmas` (`np.ndarray`): An array containing lemmas to be grouped.
- `verbose` (`bool`, optional): If `True`, prints progress at each 5% increment (default is `True`).

##### **Returns**

- Returns a `pd.DataFrame` with columns:
    - `"bow_id"`: Bag ID.
    - `"bow_name"`: Name of the bag, typically the first lemma in the bag.
    - `"lemma"`: The lemma.
    - `"similarity"`: The similarity score of the lemma with the `"bow_name"`.
    - `"threshold"`: The similarity threshold used to include the lemma in the bag.

##### **Errors and Exceptions**

- An error message is printed if an exception occurs during the execution, displaying the specific error encountered.

This function is valuable for natural language processing tasks where text data, particularly from social media, may contain various misspellings or alternative spellings of words, helping in normalizing the text for further analysis.


### SpaCy Preprocessing

The `SpacyPreprocessor` class provides advanced text preprocessing functionalities leveraging the SpaCy library, focusing on Spanish text. This class offers the capability of lemmatization, integration with a custom rule-based lemmatizer, and a comprehensive SpaCy pipeline for document preprocessing.

#### **Spanish Lemmatizer**

```python
from tidyX import SpacyPreprocessor

# Example token
token = "están"

# Applying the Spanish Lemmatizer
lemmatized_token = SpacyPreprocessor.spanish_lemmatizer(token, model)

# Printing the lemmatized token
print("Lemmatized Token:", lemmatized_token)
```

The `spanish_lemmatizer` static method is used for lemmatizing Spanish tokens using a SpaCy language model, returning a cleaned and lemmatized version of the input token.

#### **Custom Lemmatizer**

```python
import spacy
from tidyX import SpacyPreprocessor

# Loading a SpaCy Spanish model
nlp = spacy.load('es_core_news_sm')

# Adding the custom lemmatizer to the pipeline
nlp.add_pipe('custom_lemmatizer', name='lemmatizer')

# Example usage
doc = nlp("El gato está en la casa")
print("Lemmatized Text:", [token.lemma_ for token in doc])
```

The `custom_lemmatizer` is a factory function that enables integration of a custom rule-based lemmatizer for Spanish, which can be added to the SpaCy pipeline.

#### **SpaCy Pipeline**

```python
from tidyX import SpacyPreprocessor

# Example documents
documents = ["El gato está en la casa", "Los perros son animales fieles"]

# Applying the SpaCy pipeline with custom lemmatizer and extracting most common words
processed_docs, common_words = SpacyPreprocessor.spacy_pipeline(documents, custom_lemmatizer=True, num_strings=2)

# Printing the processed documents and common words
print("Processed Documents:", processed_docs)
print("Common Words:", common_words)
```

The `spacy_pipeline` static method facilitates document preprocessing through a customizable SpaCy pipeline. It handles tokenization, lemmatization, stopword removal, and the application of a custom lemmatizer. Additionally, it can return the most common words or strings in the processed documents.


### Text Visualization

The `TextVisualizer` class is integrated within the `tidyX` package, offering visualization capabilities for dependency parsing and named entities in textual data, leveraging the spaCy library's `displacy` visualizer.

#### **Dependency Parse Visualizer**

The `dependency_parse_visualizer_text` method visualizes the syntactic dependency parse or named entities within a given document, facilitating a graphical representation that aids in understanding the linguistic annotations and structure of the text.

##### **Usage**

```python
from tidyX import TextVisualizer

# Example document
document = "El gato está en la casa."

# Visualizing the dependency parse in a Jupyter environment
TextVisualizer.dependency_parse_visualizer_text(document)

# For entity visualization, set style='ent'
# TextVisualizer.dependency_parse_visualizer_text(document, style='ent')

# For usage outside of Jupyter notebooks, set jupyter=False, and it will return an HTML string.
# html = TextVisualizer.dependency_parse_visualizer_text(document, jupyter=False)
```

##### **Parameters**

- `document` (str): The input text to be visualized.
- `style` (str, optional): Determines the style of visualization - `'dep'` for dependency parse (default), `'ent'` for entities.
- `jupyter` (bool, optional): Specifies whether the visualization is intended for a Jupyter notebook (default is `True`).
- `model` (str, optional): Specifies the spaCy language model to be used (default is `'es_core_news_sm'`).

##### **Returns**

- If `jupyter` is set to `True`, the visualization is directly displayed, and the method returns `None`.
- If `jupyter` is set to `False`, the method returns a string containing the HTML representation of the visualization, which can be rendered in a web browser.

##### **Errors and Exceptions**

- Raises a `ValueError` if the `document` is empty or not a string.
- Raises a `ValueError` if an invalid style is provided.

This visualization tool is versatile and can be adapted for various NLP visualization needs, such as understanding syntactic structures or identifying named entities within texts.

## Contributing

Contributions to enhance `tidyX` are welcome! Feel free to open issues for bug reports, feature requests, or submit pull requests.

## License

MIT License
