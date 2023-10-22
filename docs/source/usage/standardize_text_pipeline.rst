Standardize Text Pipeline
-------------------------
  
The `preprocess() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.preprocess>`_ method from tidyX provides an all-encompassing solution for quickly and effectively standardizing input strings, with a particular focus on tweets. It transforms the input to lowercase, strips accents (and emojis, if specified), and removes URLs, hashtags, and certain special characters. Additionally, it offers the option to delete stopwords in a specified language, trims extra spaces, extracts mentions, and removes 'RT' prefixes from retweets.

.. code-block:: python

   from tidyX import TextPreprocessor as tp

   # Raw tweet example
   raw_tweet = "RT @user: Check out this link: https://example.com 🌍 #example 😃"

   # Applying the preprocess method
   cleaned_text = tp.preprocess(raw_tweet)

   # Printing the cleaned text
   print("Cleaned Text:", cleaned_text)

.. parsed-literal::

   Cleaned Text: check out this link

To remove English stopwords, simply add the parameters :code:`remove_stopwords=True` and :code:`language_stopwords="english"`:

.. code-block:: python

   from tidyX import TextPreprocessor as tp

   # Raw tweet example
   raw_tweet = "RT @user: Check out this link: https://example.com 🌍 #example 😃"

   # Applying the preprocess method with additional parameters
   cleaned_text = tp.preprocess(raw_tweet, remove_stopwords=True, language_stopwords="english")

   # Printing the cleaned text
   print("Cleaned Text:", cleaned_text)

.. parsed-literal::

   Cleaned Text: check link

For a more detailed explanation of the customizable steps of the function, visit the official `preprocess() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.preprocess>`_ documentation.

