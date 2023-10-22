Stemming and Lemmatizing
-------------------------

One of the foundational steps in preparing text for NLP applications is bringing words to a common base or root. This library provides both `stemmer() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextNormalization.html#tidyX.text_normalization.TextNormalization.stemmer>`_ and `lemmatizer() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextNormalization.html#tidyX.text_normalization.TextNormalization.lemmatizer>`_ functions to perform this task across various languages.

Make sure you have the necessary dependencies installed. If you plan on lemmatizing, you'll need `spaCy` along with the appropriate language models. For Spanish lemmatization, we recommend downloading the `es_core_news_sm` model:

.. code-block:: bash

   python -m spacy download es_core_news_sm

For English lemmatization, we suggest the `en_core_web_sm` model:

.. code-block:: bash

   python -m spacy download en_core_web_sm 

   
.. code-block:: python

   from tidyX import TextNormalization as tn
   import spacy

   # Input token to lemmatize in English
   token = "running"  
   print("Original Token:", token)

   # Load spacy's model for English
   model_en = spacy.load('en_core_news_sm')

   # Apply lemmatizer function to lemmatize the token
   lemmatized_token = tn.lemmatizer(token=token, model=model_en)
   print("Lemmatized Token:", lemmatized_token)

.. parsed-literal::

   Original Token: running
   Lemmatized Token: run

.. code-block:: python

   from tidyX import TextNormalization as tn
   import spacy

   # Input token to lemmatize in Spanish
   token = "corriendo"  
   print("Original Token:", token)

   # Load spacy's model for Spanish
   model_es = spacy.load('es_core_news_sm')

   # Apply lemmatizer function to lemmatize the token
   lemmatized_token = tn.lemmatizer(token=token, model=model_es)
   print("Lemmatized Token:", lemmatized_token)

.. parsed-literal::

   Original Token: corriendo
   Lemmatized Token: correr

For stemming, we'll use the Snowball stemmer. Here's how you can use the function:

.. code-block:: python

   from tidyX import TextNormalization as tn
   from nltk.stem import SnowballStemmer

   # Input token to stem in English
   token_en = "running"  
   print("Original Token:", token_en)

   # Apply stemmer function to stem the token in English
   stemmed_token_en = tn.stemmer(token=token_en, language="english")
   print("Stemmed Token:", stemmed_token_en)

.. parsed-literal::

   Original Token: running
   Stemmed Token: run

.. code-block:: python

   # Input token to stem in Spanish
   token_es = "corriendo"  
   print("Original Token:", token_es)

   # Apply stemmer function to stem the token in Spanish
   stemmed_token_es = tn.stemmer(token=token_es)
   print("Stemmed Token:", stemmed_token_es)

.. parsed-literal::

   Original Token: corriendo
   Stemmed Token: corr

Note that these functions are built to receive a single token as input; however, most applications require normalizing all the words within a text. To do this efficiently, refer to the `Stemming and Lemmatizing Texts Efficiently <https://tidyx.readthedocs.io/en/latest/tutorials/stemming_and_lemmatizing_efficiently.html>`_ tutorial.
