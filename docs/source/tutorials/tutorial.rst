tidyX examples
==============

.. code:: ipython3

    # pip install tidyX==1.6.7

.. code:: ipython3

    !pip show tidyX


.. parsed-literal::

    Name: tidyX
    Version: 1.6.7
    Summary: Python package to clean raw tweets for ML applications
    Home-page: 
    Author: Lucas Gómez Tobón, Jose Fernando Barrera
    Author-email: lucasgomeztobon@gmail.com, jf.barrera10@uniandes.edu.co
    License: MIT
    Location: c:\users\lucas\anaconda3\envs\bx\lib\site-packages
    Requires: emoji, nltk, numpy, pandas, regex, spacy, thefuzz, Unidecode
    Required-by: 
    

.. code:: ipython3

    from tidyX import TextPreprocessor as tp
    from tidyX import TextNormalization as tn
    from tidyX import TextVisualizer as tv

Stemming and Lemmatizing Texts Efficiently
------------------------------------------

The ``stemmer()`` and ``lemmatizer()`` functions each accept a single
token as input. Thus, if we aim to normalize an entire text or a corpus,
we would need to iterate over each token in the string using these
functions. This approach might be inefficient, especially if the input
contains repeated words.

This tutorial demonstrates how to utilize the ``unnest_tokens()``
function to apply normalization functions just once for every unique
word.

.. code:: ipython3

    # First, load a dataframe containing 1000 tweets from Colombia discussing Venezuela.
    tweets = tp.load_data(file = "spanish")
    tweets.head()




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



.. code:: ipython3

    # Firstly we would clean the text easily using our preprocess function
    tweets['clean'] = tweets['Tweet'].apply(lambda x: tp.preprocess(x, 
                                                                    delete_emojis = False, 
                                                                    remove_stopwords = True, 
                                                                    language_stopwords = "spanish"))
    tweets.head()




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
          <th>clean</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>regala moneda pa cafe venezolano no tuitero ah...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>RT @CriptoNoticias Banco venezolano activa ser...</td>
          <td>banco venezolano activa servicio usuarios crip...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Capturado venezolano que asesinó a comerciante...</td>
          <td>capturado venezolano asesino comerciante merca...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>RT @PersoneriaVpar @PersoneriaVpar acompaña al...</td>
          <td>acompa grupo especial migratorio cesar reunion...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Bueno ya sacaron la carta de "amenaza de atent...</td>
          <td>bueno sacaron carta amenaza atentado president...</td>
        </tr>
      </tbody>
    </table>
    </div>



In this step, we will utilize the ``unnest_token()`` function to divide
each tweet into multiple rows, assigning one token to each row. This
structure allows us to aggregate identical terms, thereby creating an
auxiliary dataframe that acts as a dictionary for lemmas or stems.

.. code:: ipython3

    dictionary_normalization = tp.unnest_tokens(df = tweets.copy(), input_column = "clean", id_col = None, unique = True)
    dictionary_normalization




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
          <th>clean</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td></td>
          <td>246</td>
        </tr>
        <tr>
          <th>1</th>
          <td>abajo</td>
          <td>352, 577</td>
        </tr>
        <tr>
          <th>2</th>
          <td>abandonar</td>
          <td>337, 509</td>
        </tr>
        <tr>
          <th>3</th>
          <td>abarrotarse</td>
          <td>993</td>
        </tr>
        <tr>
          <th>4</th>
          <td>abiertos</td>
          <td>72</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>5878</th>
          <td>🤪</td>
          <td>519</td>
        </tr>
        <tr>
          <th>5879</th>
          <td>🤬</td>
          <td>483, 520, 908, 908</td>
        </tr>
        <tr>
          <th>5880</th>
          <td>🤯</td>
          <td>615</td>
        </tr>
        <tr>
          <th>5881</th>
          <td>🤷</td>
          <td>482, 736, 841, 947, 947, 947</td>
        </tr>
        <tr>
          <th>5882</th>
          <td>🥺</td>
          <td>833, 851</td>
        </tr>
      </tbody>
    </table>
    <p>5883 rows × 2 columns</p>
    </div>



Note that the ``id`` column represents the indices of the tweets that
contain each token from the ``clean`` column. Now we can proceed using
the ``stemmer()`` and ``lemmatizer()`` functions to create new columns
of ``dictionary_normalization``

.. code:: ipython3

    # Apply spanish_lemmatizer function to lemmatize the token
    dictionary_normalization["stemm"] = dictionary_normalization["clean"].apply(lambda x: tn.stemmer(token = x, language = "spanish"))

Don’t forget to download the corresponding SpaCy model for
lemmatization. For Spanish lemmatization, we suggest the
``es_core_news_sm`` model:

.. code:: bash

   !python -m spacy download es_core_news_sm   

For English lemmatization, we suggest the ``en_core_web_sm`` model:

.. code:: bash

   !python -m spacy download en_core_web_sm 

To see a full list of available models for different languages, visit
`Spacy’s documentation <https://spacy.io/models/>`__

.. code:: ipython3

    import spacy
    
    # Load model
    model_es = spacy.load("es_core_news_sm")
    
    # Apply lemmatizer function to lemmatize the token
    dictionary_normalization["lemma"] = dictionary_normalization["clean"].apply(lambda x: tn.lemmatizer(token = x, model = model_es))
    
    # Lemmatizing could produce stopwords, therefore we applied remove_words function
    dictionary_normalization["lemma"] = dictionary_normalization["lemma"].apply(lambda x: tp.remove_words(x, remove_stopwords = True, language = "spanish"))
    
    dictionary_normalization




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
          <th>clean</th>
          <th>id</th>
          <th>stemm</th>
          <th>lemma</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td></td>
          <td>246</td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>1</th>
          <td>abajo</td>
          <td>352, 577</td>
          <td>abaj</td>
          <td>abajo</td>
        </tr>
        <tr>
          <th>2</th>
          <td>abandonar</td>
          <td>337, 509</td>
          <td>abandon</td>
          <td>abandonar</td>
        </tr>
        <tr>
          <th>3</th>
          <td>abarrotarse</td>
          <td>993</td>
          <td>abarrot</td>
          <td>abarrotar</td>
        </tr>
        <tr>
          <th>4</th>
          <td>abiertos</td>
          <td>72</td>
          <td>abiert</td>
          <td>abierto</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>5878</th>
          <td>🤪</td>
          <td>519</td>
          <td>🤪</td>
          <td>🤪</td>
        </tr>
        <tr>
          <th>5879</th>
          <td>🤬</td>
          <td>483, 520, 908, 908</td>
          <td>🤬</td>
          <td>🤬</td>
        </tr>
        <tr>
          <th>5880</th>
          <td>🤯</td>
          <td>615</td>
          <td>🤯</td>
          <td>🤯</td>
        </tr>
        <tr>
          <th>5881</th>
          <td>🤷</td>
          <td>482, 736, 841, 947, 947, 947</td>
          <td>🤷</td>
          <td>🤷</td>
        </tr>
        <tr>
          <th>5882</th>
          <td>🥺</td>
          <td>833, 851</td>
          <td>🥺</td>
          <td>🥺</td>
        </tr>
      </tbody>
    </table>
    <p>5883 rows × 4 columns</p>
    </div>



To rebuild our original tweets we will use again ``unnest_tokens``
function

.. code:: ipython3

    tweets_long = tp.unnest_tokens(df = tweets.copy(), input_column = "clean", id_col = None, unique = False)
    tweets_long




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
          <th>clean</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>regala</td>
          <td>0</td>
        </tr>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>moneda</td>
          <td>0</td>
        </tr>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>pa</td>
          <td>0</td>
        </tr>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>cafe</td>
          <td>0</td>
        </tr>
        <tr>
          <th>0</th>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>venezolano</td>
          <td>0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>999</th>
          <td>RT infopresidencia: "Sin lugar a dudas hay uno...</td>
          <td>recibido</td>
          <td>999</td>
        </tr>
        <tr>
          <th>999</th>
          <td>RT infopresidencia: "Sin lugar a dudas hay uno...</td>
          <td>cerca</td>
          <td>999</td>
        </tr>
        <tr>
          <th>999</th>
          <td>RT infopresidencia: "Sin lugar a dudas hay uno...</td>
          <td>venezolanos</td>
          <td>999</td>
        </tr>
        <tr>
          <th>999</th>
          <td>RT infopresidencia: "Sin lugar a dudas hay uno...</td>
          <td>presidente</td>
          <td>999</td>
        </tr>
        <tr>
          <th>999</th>
          <td>RT infopresidencia: "Sin lugar a dudas hay uno...</td>
          <td>i</td>
          <td>999</td>
        </tr>
      </tbody>
    </table>
    <p>13557 rows × 3 columns</p>
    </div>



.. code:: ipython3

    tweets_normalized = tweets_long \
        .merge(dictionary_normalization, how = "left", on = "clean") \
            .groupby(["id_x", "Tweet"])[["lemma", "stemm"]] \
                .agg(lambda x: " ".join(x)) \
                    .reset_index()
    tweets_normalized.head()




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
          <th>stemm</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>regalar moneda pa cafar venezolano  tuitero ah...</td>
          <td>regal moned pa caf venezolan no tuiter ah 😂 👋</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>RT @CriptoNoticias Banco venezolano activa ser...</td>
          <td>banco venezolano activo servicio usuario cript...</td>
          <td>banc venezolan activ servici usuari criptomoned</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>Capturado venezolano que asesinó a comerciante...</td>
          <td>capturado venezolano asesino comerciante merca...</td>
          <td>captur venezolan asesin comerci merc public</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>RT @PersoneriaVpar @PersoneriaVpar acompaña al...</td>
          <td>acompa grupo especial migratorio cesar reunion...</td>
          <td>acomp grup especial migratori ces reunion real...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>Bueno ya sacaron la carta de "amenaza de atent...</td>
          <td>bueno sacar cartar amenazar atentado president...</td>
          <td>buen sac cart amenaz atent president duqu func...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    for i in range(3):
        print("-"*50)
        print("Example", i + 1)
        print("Original tweet:", tweets_normalized.loc[i, "Tweet"])
        print("Lemmatized tweet:", tweets_normalized.loc[i, "lemma"])
        print("Stemmed tweet:", tweets_normalized.loc[i, "stemm"])


.. parsed-literal::

    --------------------------------------------------
    Example 1
    Original tweet: RT @emilsen_manozca ¿Me regala una moneda pa un café? -¿Eres venezolano? Noo! Tuitero. -Ahhh 😂😂😂👋
    Lemmatized tweet: regalar moneda pa cafar venezolano  tuitero ah 😂 👋
    Stemmed tweet: regal moned pa caf venezolan no tuiter ah 😂 👋
    --------------------------------------------------
    Example 2
    Original tweet: RT @CriptoNoticias Banco venezolano activa servicio para usuarios de criptomonedas #ServiciosFinancieros https://t.co/1r2rZIUdlo
    Lemmatized tweet: banco venezolano activo servicio usuario criptomoneda
    Stemmed tweet: banc venezolan activ servici usuari criptomoned
    --------------------------------------------------
    Example 3
    Original tweet: Capturado venezolano que asesinó a comerciante del Mercado Público https://t.co/XrmWKVYMR8 https://t.co/CfMLaB25jI
    Lemmatized tweet: capturado venezolano asesino comerciante mercado publico
    Stemmed tweet: captur venezolan asesin comerci merc public
    

Tutorial: Word Cloud
--------------------

.. code:: ipython3

    import os
    import pandas as pd
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import spacy
     
    os.getcwd()

.. code:: ipython3

    tweets = pd.read_excel(r"../../../data/Tweets sobre venezuela.xlsx")
    tweets.head()

.. code:: ipython3

    # Combine all documents into a single string
    text = " ".join(doc for doc in tweets['Snippet'])
    
    # Generate a word cloud image
    wordcloud = WordCloud(background_color = "white", width = 800, height = 400).generate(text)
    
    # Display the generated image
    plt.figure(figsize=(10, 5))
    plt.title("WordCloud before tidyX")
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off");

.. code:: ipython3

    tweets['clean'] = tweets['Snippet'].apply(lambda x: tp.preprocess(x, delete_emojis = False, extract = False,
                                                                      remove_stopwords = True))
    tweets

.. code:: ipython3

    token_df = tp.unnest_tokens(df = tweets.copy(), input_column = "clean", id_col = None, unique = True)
    token_df

.. code:: ipython3

    # Load spacy's model
    model = spacy.load('es_core_news_lg')

.. code:: ipython3

    # Apply spanish_lemmatizer function to lemmatize the token
    token_df["lemma"] = token_df["clean"].apply(lambda x: tn.lemmatizer(token = x, model = model))
    token_df

.. code:: ipython3

    token_df["lemma"] = token_df["lemma"].apply(lambda x: tp.remove_words(x, remove_stopwords = True))
    token_df = token_df[["clean", "lemma"]]
    token_df

.. code:: ipython3

    tweets_long = tp.unnest_tokens(df = tweets.copy(), input_column = "clean", id_col = None, unique = False)
    tweets_long 

.. code:: ipython3

    tweets_clean2 = tweets_long.merge(token_df, how = "left", on = "clean").groupby(["Snippet", "id"])["lemma"].agg(lambda x: " ".join(x)).reset_index()
    tweets_clean2

.. code:: ipython3

    tweets_clean2['lemma'] = tweets_clean2['lemma'].apply(lambda x: tp.remove_extra_spaces(x))

.. code:: ipython3

    # Combine all documents into a single string
    text = " ".join(doc for doc in tweets_clean2['lemma'])
    
    # Generate a word cloud image
    wordcloud = WordCloud(background_color = "white", width = 800, height = 400).generate(text)
    
    # Display the generated image
    plt.figure(figsize=(10, 5))
    plt.title("WordCloud after tidyX")
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off");
