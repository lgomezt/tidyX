Stemming and Lemmatizing
-------------------------

One of the foundational steps in preparing text for NLP applications is bringing words to a common base or root. This library provides both `stemmer()` and `lemmatizer()` functions to perform this task across various languages.

Make sure you have the necessary dependencies installed. If you plan on lemmatizing, you'll need `spaCy` along with the appropriate language models. For Spanish lemmatization, we recommend downloading the `es_core_web_sm` model:

.. code-block:: bash

   python -m spacy download es_core_web_sm   

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

Note that these functions are built to receive a single token as input; however, most applications require normalizing all the words within a text. To do this efficiently, refer to the `Stemming and Lemmatizing Texts Efficiently <some_link>`_ tutorial.
