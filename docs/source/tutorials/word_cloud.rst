Word Cloud
==========

This tutorial demonstrates how to create a word cloud from a corpus of tweets written in Spanish.

First, set up the necessary modules from the package:

.. code-block:: python

    # Import modules from the tidyX package for text preprocessing, normalization, and visualization
    from tidyX import TextPreprocessor as tp
    from tidyX import TextNormalization as tn
    from tidyX import TextVisualizer as tv

    # Import auxiliary libraries for data manipulation and visualization
    import os
    import pandas as pd
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import spacy  # Natural Language Processing library

    # Load a dataframe that contains 1000 tweets from Colombia discussing Venezuela
    tweets = tp.load_data(file="spanish")
    tweets.head()  # Display the first few rows of the dataframe

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Tweet</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>RT @CriptoNoticias Banco venezolano activa ser...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Capturado venezolano que asesinó a comerciante...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>RT @PersoneriaVpar @PersoneriaVpar acompaña al...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Bueno ya sacaron la carta de "amenaza de atent...</td>
        </tr>
      </tbody>
    </table>
    </div>

For illustrative purposes, generate a word cloud without preprocessing the text:

.. code-block:: python

    # Combine all individual tweets into one large text string
    text = " ".join(doc for doc in tweets['Tweet'])
    
    # Generate a word cloud from the combined text
    wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)
    
    # Visualize and display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.title("WordCloud before tidyX")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # Hide the axis for better aesthetics

.. image:: wordcloud_before_tidyX.png
   :alt: before_tidyX
   :width: 800px

Next, preprocess the tweets using techniques outlined in the `Stemming and Lemmatizing Texts Efficiently <https://tidyx.readthedocs.io/en/latest/tutorials/stemming_and_lemmatizing_efficiently.html>`_ tutorial:

.. code-block:: python

    # Clean the text: remove emojis, stopwords, and apply other preprocessing steps
    tweets['clean'] = tweets['Tweet'].apply(lambda x: tp.preprocess(x, 
                                                                    delete_emojis=False, 
                                                                    remove_stopwords=True, 
                                                                    language_stopwords="spanish"))
    
    # Tokenize the cleaned text to create a dictionary for normalization
    dictionary_normalization = tp.unnest_tokens(df=tweets.copy(), input_column="clean", id_col=None, unique=True)
    
    # Load the Spanish language model from spacy for lemmatization
    model_es = spacy.load("es_core_news_sm")
    
    # Apply lemmatization to the tokens to reduce words to their base form
    dictionary_normalization["lemma"] = dictionary_normalization["clean"].apply(lambda x: tn.lemmatizer(token=x, model=model_es))
    
    # Remove any stopwords introduced after lemmatization
    dictionary_normalization["lemma"] = dictionary_normalization["lemma"].apply(lambda x: tp.remove_words(x, remove_stopwords=True, language="spanish"))
    
    # Reconstruct the original tweets using lemmatized tokens
    tweets_long = tp.unnest_tokens(df=tweets.copy(), input_column="clean", id_col=None, unique=False)
    tweets_normalized = tweets_long \
        .merge(dictionary_normalization, how="left", on="clean") \
            .groupby(["id_x", "Tweet"])["lemma"] \
                .agg(lambda x: " ".join(x)) \
                    .reset_index()

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id_x</th>
          <th>Tweet</th>
          <th>lemma</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>regalar moneda pa cafar venezolano  tuitero ah...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>RT @CriptoNoticias Banco venezolano activa ser...</td>
          <td>banco venezolano activo servicio usuario cript...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>Capturado venezolano que asesinó a comerciante...</td>
          <td>capturado venezolano asesino comerciante merca...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>RT @PersoneriaVpar @PersoneriaVpar acompaña al...</td>
          <td>acompa grupo especial migratorio cesar reunion...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>Bueno ya sacaron la carta de "amenaza de atent...</td>
          <td>bueno sacar cartar amenazar atentado president...</td>
        </tr>
      </tbody>
    </table>
    </div>

.. code-block:: python

    for i in range(3):
        print("-"*50)
        print("Example", i + 1)
        print("Original tweet:", tweets_normalized.loc[i, "Tweet"])
        print("Lemmatized tweet:", tweets_normalized.loc[i, "lemma"])

.. parsed-literal::

    --------------------------------------------------
    Example 1
    Original tweet: RT @emilsen_manozca ¿Me regala una moneda pa un café? -¿Eres venezolano? Noo! Tuitero. -Ahhh 😂😂😂👋
    Lemmatized tweet: regalar moneda pa cafar venezolano  tuitero ah 😂 👋
    --------------------------------------------------
    Example 2
    Original tweet: RT @CriptoNoticias Banco venezolano activa servicio para usuarios de criptomonedas #ServiciosFinancieros https://t.co/1r2rZIUdlo
    Lemmatized tweet: banco venezolano activo servicio usuario criptomoneda
    --------------------------------------------------
    Example 3
    Original tweet: Capturado venezolano que asesinó a comerciante del Mercado Público https://t.co/XrmWKVYMR8 https://t.co/CfMLaB25jI
    Lemmatized tweet: capturado venezolano asesino comerciante mercado publico
    
Lastly, generate a word cloud using the preprocessed and lemmatized text:

.. code-block:: python
  
    # Combine all lemmatized tweets into one large text string
    text = " ".join(doc for doc in tweets_normalized['lemma'])
    
    # Generate a word cloud from the lemmatized text
    wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)
    
    # Visualize and display the new word cloud
    plt.figure(figsize=(10, 5))
    plt.title("WordCloud after tidyX")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # Hide the axis for better aesthetics

.. image:: wordcloud_after_tidyX.png
   :alt: after_tidyX
   :width: 800px
