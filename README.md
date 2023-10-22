# tidyX

`tidyX` is a Python package designed for cleaning and preprocessing text for machine learning applications, **especially for text written in Spanish and originating from social networks.** This library provides a complete pipeline to remove unwanted characters, normalize text, group similar terms, etc. to facilitate NLP applications.

**To deep dive in the package visit our [website](https://tidyx.readthedocs.io/en/latest/)

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

To see a full list of available models for different languages, visit [Spacy's documentation](https://spacy.io/models/).


## Features

- **Standardize Text Pipeline**: The `preprocess()` method provides an all-encompassing solution for quickly and effectively standardizing input strings, with a particular focus on tweets. It transforms the input to lowercase, strips accents (and emojis, if specified), and removes URLs, hashtags, and certain special characters. Additionally, it offers the option to delete stopwords in a specified language, trims extra spaces, extracts mentions, and removes 'RT' prefixes from retweets.

```python
from tidyX import TextPreprocessor as tp

# Raw tweet example
raw_tweet = "RT @user: Check out this link: https://example.com 🌍 #example 😃"

# Applying the preprocess method
cleaned_text = tp.preprocess(raw_tweet)

# Printing the cleaned text
print("Cleaned Text:", cleaned_text)
```

**Output**:
```
Cleaned Text: check out this link
```

To remove English stopwords, simply add the parameters `remove_stopwords=True` and `language_stopwords="english"`:

```python
from tidyX import TextPreprocessor as tp

# Raw tweet example
raw_tweet = "RT @user: Check out this link: https://example.com 🌍 #example 😃"

# Applying the preprocess method with additional parameters
cleaned_text = tp.preprocess(raw_tweet, remove_stopwords=True, language_stopwords="english")

# Printing the cleaned text
print("Cleaned Text:", cleaned_text)
```

**Output**:
```
Cleaned Text: check link
```

For a more detailed explanation of the customizable steps of the function, visit the official [preprocess() documentation](https://tidyx.readthedocs.io/en/latest/api/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.preprocess).


- **Stemming and Lemmatizing**: One of the foundational steps in preparing text for NLP applications is bringing words to a common base or root. This library provides both `stemmer()` and `lemmatizer()` functions to perform this task across various languages.
- **Group similar terms**: When working with a corpus sourced from social networks, it's common to encounter texts with grammatical errors or words that aren't formally included in dictionaries. These irregularities can pose challenges when creating Term Frequency matrices for NLP algorithms. To address this, we developed the [`create_bol()`](https://tidyx.readthedocs.io/en/latest/examples/tutorial.html#create-bol) function, which allows you to create specific bags of terms to cluster related terms.
- **Remove unwanted elements**: such as special characters, accents, emojis, urls, tweeter mentions, among others.
- **Dependency Parsing Visualization**: Incorporates visualization tools that enable the display of dependency parses, facilitating linguistic analysis and feature engineering.
- **Much more!**

## Contributing

Contributions to enhance `tidyX` are welcome! Feel free to open issues for bug reports, feature requests, or submit pull requests.
