Topic Modelling
================

Introduction
--------------

In the age of social media, Twitter has become a fertile ground for data mining, sentiment analysis, and various other natural language processing (NLP) tasks. However, dealing with Spanish tweets adds another layer of complexity due to language-specific nuances, slang, abbreviations, and other colloquial expressions. **tidyX** aims to streamline the preprocessing pipeline for Spanish tweets, making them ready for various NLP tasks such as text classification, topic modeling, sentiment analysis, and more. In this tutorial, we will focus on a classification task based on Topic Modelling, showing preprocessing, modeling and results with real data snippets.

Context
-------

Using data provided by `Barómetro de Xenofobia <https://barometrodexenofobia.org/>`__, a non-profit organization that quantifies the amount of hate speech against migrants on social media, we aim to classify the overall conversation related to migrants. This is a **common NLP task** that involves preprocessing poorly-written social media posts. Subsequently, these processed posts are fed into an unsupervised Topic Classification Model (LDA) to identify an optimal number of cluster topics. This helps reveal the main discussion points concerning Venezuelan migrants in Colombia.

.. code-block:: python

    # Import TidyX and other libraries. 
    from tidyX import TextPreprocessor as tp
    from tidyX import TextNormalization as tn

    # Import other libraries needed in this tutorial
    import pandas as pd
    import os
    import gensim
    from gensim import corpora 
    from gensim.models import CoherenceModel
    import tqdm 
    import numpy as np
    import itertools
    from collections import Counter
    import pprint
    import pyLDAvis
    pyLDAvis.enable_notebook()
    import pyLDAvis.gensim_models
    import spacy

    # Load a dataframe that contains 1000 tweets from Colombia discussing Venezuela
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
          <th>Snippet</th>
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
        <tr>
          <th>5</th>
          <td>@IvanDuque es muy bueno que se le dé respaldo ...</td>
        </tr>
        <tr>
          <th>6</th>
          <td>RT @RafaelG10099924 @mluciaramirez @Eganbernal...</td>
        </tr>
        <tr>
          <th>7</th>
          <td>#ParaVenezuelaPropongo que se levante el bloqu...</td>
        </tr>
        <tr>
          <th>8</th>
          <td>RT @geoduque La diferencia entre la preocupaci...</td>
        </tr>
        <tr>
          <th>9</th>
          <td>RT @PanamericanaTV ¡No le abrió la puerta de s...</td>
        </tr>
      </tbody>
    </table>
    </div>


Preprocessing Tweets
--------------------

We will then use `preprocess() <https://tidyx.readthedocs.io/en/latest/usage/standardize_text_pipeline.html>`_ function to clean the sample and prepare it for modelling

.. code-block:: python

    cleaning_process = lambda x: tp.preprocess(x, delete_emojis = True, extract = False, remove_stopwords = True, language_stopwords = "spanish")
    tweets['Clean_tweets'] = tweets['Tweet'].apply(cleaning_process)


Here is a random sample of the before and after with specific Tweets

.. code-block:: python

    # You can change the random_state for different samples
    sample_tweets = tweets.sample(5, random_state = 1)  

    print("Before and After Text Cleaning:")
    print('-' * 40)
    for index, row in sample_tweets.iterrows():
        print(f"Original: {row['Tweet']}")
        print(f"Cleaned: {row['Clean_tweets']}")
        print('-' * 40)


.. parsed-literal::

    Before and After Text Cleaning:
    ----------------------------------------
    Original: Antes el pasaporte venezolano permitía la entrada en en sinfín de países del mundo. Hoy cada día estamos más limitados gracias al socialismo del siglo 21. Hasta Cuba, que saquea a Venezuela, nos impone una visa. #PeroTodoTieneSuFinal
    Cleaned: pasaporte venezolano permitia entrada sinfin paises mundo hoy cada dia limitados gracias socialismo siglo cuba saquea venezuela impone visa
    ----------------------------------------
    Original: @VickyDavilaH Bueno y si @AlvaroUribeVel se proclama presidente de una vez por todas y nombra a @IvanDuque ministro de guerra y lo deja que solito libere al pueblo venezolano, ¿será que le prestan atención a la grave crisis que vive el Chocó, que parece que solo cuentan con el Esmad ?
    Cleaned: bueno proclama presidente vez todas nombra ministro guerra deja solito libere pueblo venezolano prestan atencion grave crisis vive choco parece solo cuentan esmad
    ----------------------------------------
    Original: @zonacero Nomás quieren Telesur y Venezolana de Televisión, super imparcialicimos.
    Cleaned: nomas quieren telesur venezolana television super imparcialicimos
    ----------------------------------------
    Original: RT @XiomaryUrbaez Sr @jguaido yo, venezolana y residente en el país, SÍ QUIERO INTERVENCIÓN. Le agradezco que sin haber hecho una consulta pública sobre algo tan importante, no hable por mí. Gracias.
    Cleaned: sr venezolana residente pais quiero intervencion agradezco haber hecho consulta publica tan importante hable gracias
    ----------------------------------------
    Original: Y también las grandes masas de venezolanos queriendo refugiarse en Colombia, de verdad que esto es una gran insensatez descarada y cruel, porque todo está premeditadamente calculado.
    Cleaned: grandes masas venezolanos queriendo refugiarse colombia verdad gran insensatez descarada cruel premeditadamente calculado
    ----------------------------------------
    

Tokenize and lemmatize tweets in the dataset
---------------------------------------------

We use `unnest_token() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.unnest_tokens>`_ function to divide each tweet into multiple rows, assigning one token to each row. This structure allows us to aggregate identical terms, thereby creating an auxiliary dataframe that acts as a dictionary for lemmas.

We want an iterable of lemmatized non-stopword tokens in order to recreate a cleaner version of the tweet. In order to achieve that, we call `tn.lemmatizer() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextNormalization.html#tidyX.text_normalization.TextNormalization.lemmatizer>`_ returning an original base form of a token in a specific language structure.

.. code-block:: python

    # load a spaCy model, depending on language, out-of-the-box
    model_es = spacy.load("es_core_news_sm") # depends on your needs. Please visit: https://spacy.io/models
    # Create a dictionary of tokens to lemmatize
    word_dict = tp.unnest_tokens(df = tweets.copy(), input_column = 'Clean_tweets', id_col = None, unique = True)
    # Lemmatize the tokens
    word_dict["lemmatized_tweets"] = word_dict["Clean_tweets"].apply(lambda x: tn.lemmatizer(token = x, model = model_es))
    # Rebuild the tweets using the lemmatized tokens
    rebuild_tweets = tp.unnest_tokens(df = tweets.copy(), input_column = "Clean_tweets", id_col = None, unique = False)
    tokenized_cleaned_tweets = rebuild_tweets \
        .merge(word_dict, how = "left", on = "Clean_tweets") \
            .groupby(["id_x", "Snippet"])[["lemmatized_tweets"]] \
                .agg(lambda x: " ".join(x)) \
                    .reset_index()
    tokenized_cleaned_tweets.head(3)

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
          <th>Snippet</th>
          <th>lemmatized_tweets</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>RT @emilsen_manozca ¿Me regala una moneda pa u...</td>
          <td>regalar moneda pa cafe venezolano no tuitero ah</td>
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
      </tbody>
    </table>
    </div>


Here is a random sample of the before and after with specific Tweets

.. code-block:: python

    tweets['lemmatized_tweets'] = tokenized_cleaned_tweets['lemmatized_tweets']
    sample_tweets = tweets.sample(5, random_state=1)  # You can change the random_state for different samples
    print("Before and After Text Cleaning:")
    print('-' * 40)
    for index, row in sample_tweets.iterrows():
        print(f"Original: {row['Snippet']}")
        print(f"Cleaned: {row['lemmatized_tweets']}")
        print('-' * 40)


.. parsed-literal::

    Before and After Text Cleaning:
    ----------------------------------------
    Original: Antes el pasaporte venezolano permitía la entrada en en sinfín de países del mundo. Hoy cada día estamos más limitados gracias al socialismo del siglo 21. Hasta Cuba, que saquea a Venezuela, nos impone una visa. #PeroTodoTieneSuFinal
    Cleaned: pasaporte venezolano permitia entrada sinfin pais mundo hoy cada diar limitado gracias socialismo siglo cuba saquea venezuela imponer vis
    ----------------------------------------
    Original: @VickyDavilaH Bueno y si @AlvaroUribeVel se proclama presidente de una vez por todas y nombra a @IvanDuque ministro de guerra y lo deja que solito libere al pueblo venezolano, ¿será que le prestan atención a la grave crisis que vive el Chocó, que parece que solo cuentan con el Esmad ?
    Cleaned: bueno proclamar presidente vez todo nombra ministro guerra dejar solitir liberar pueblo venezolano prestar atencion grave crisis vivir choco parecer solo contar esmad
    ----------------------------------------
    Original: @zonacero Nomás quieren Telesur y Venezolana de Televisión, super imparcialicimos.
    Cleaned: noma querer telesur venezolano television super imparcialicir
    ----------------------------------------
    Original: RT @XiomaryUrbaez Sr @jguaido yo, venezolana y residente en el país, SÍ QUIERO INTERVENCIÓN. Le agradezco que sin haber hecho una consulta pública sobre algo tan importante, no hable por mí. Gracias.
    Cleaned: sr venezolano residente pai querer intervencion agradecer haber hecho consulta publicar tanto importante hablar gracias
    ----------------------------------------
    Original: Y también las grandes masas de venezolanos queriendo refugiarse en Colombia, de verdad que esto es una gran insensatez descarada y cruel, porque todo está premeditadamente calculado.
    Cleaned: grande masa venezolano querer refugiar el colombia verdad gran insensatez descarado cruel premeditadamente calculado
    ----------------------------------------
    

Seemingly used words and social media bad writting addressing
--------------------------------------------------------------

May you saw in the previous proccesed tweets that there are seemingly used or Out-of-Vocabulary (OOV) words that became evident after processing and cleaning the tweets showed. This words can be a result of bad spelling, common in social media, abbreviations, or other language rules.

Here we propose a method to handle this limitations, some research related to this topic establishes local solutions to this condition, we invite the user to try this approach and also find some other resources to proccess the resulted lemmas. Some additional research to handle OOV words can be found in:

1. `FastText <https://github.com/facebookresearch/fastText>`__
2. `Kaggle NER Bi-LSTM <https://www.kaggle.com/code/jatinmittal0001/ner-bi-lstm-dealing-with-oov-words>`__
3. `Contextual Spell Check <https://github.com/R1j1t/contextualSpellCheck>`__

We use our `create_bol() <https://tidyx.readthedocs.io/en/latest/usage/group_similar_terms.html>`_ function to find distances between lemmas, we are based on the premise that seemingly used lemmas ar far away from the original corpus and don't have a big apperance on it. 

.. code-block:: python

    # We take our lemmatized tweets and create a list of lists for the bag of lemmas
    flattened_list = list(itertools.chain.from_iterable(tokenized_cleaned_tweets['lemmatized_tweets'].str.split(" ")))
    # Now we count the number of times each lemma appears in the list and sort the list in descending order
    word_count = Counter(flattened_list)
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    sorted_words_only = [word for word, count in sorted_words]
    numpy_array = np.array(sorted_words_only)
    # Now we create our bag of lemmas
    bol_df = tp.create_bol(numpy_array, verbose=True)
    bol_df.head(10)


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
          <th>bow_id</th>
          <th>bow_name</th>
          <th>lemma</th>
          <th>similarity</th>
          <th>threshold</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>venezolano</td>
          <td>venezolano</td>
          <td>100</td>
          <td>85</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>venezolano</td>
          <td>venezolana</td>
          <td>90</td>
          <td>85</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>venezolano</td>
          <td>venezolan</td>
          <td>95</td>
          <td>85</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>venezolano</td>
          <td>venezolanado</td>
          <td>91</td>
          <td>85</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>el</td>
          <td>el</td>
          <td>100</td>
          <td>88</td>
        </tr>
        <tr>
          <th>5</th>
          <td>3</td>
          <td>pai</td>
          <td>pai</td>
          <td>100</td>
          <td>87</td>
        </tr>
        <tr>
          <th>6</th>
          <td>4</td>
          <td>colombia</td>
          <td>colombia</td>
          <td>100</td>
          <td>85</td>
        </tr>
        <tr>
          <th>7</th>
          <td>4</td>
          <td>colombia</td>
          <td>colombiano</td>
          <td>89</td>
          <td>85</td>
        </tr>
        <tr>
          <th>8</th>
          <td>5</td>
          <td>hacer</td>
          <td>hacer</td>
          <td>100</td>
          <td>86</td>
        </tr>
        <tr>
          <th>9</th>
          <td>6</td>
          <td>ser</td>
          <td>ser</td>
          <td>100</td>
          <td>87</td>
        </tr>
      </tbody>
    </table>
    </div>


Now we want to select a specific subset of words that does not include our probable OOV or NEW words in the text processing. We will replace words using 85% confidence treshold soo we can infer what was intended to be written.

.. code-block:: python

    # Replace each lemma in the original list of lists with its bow_name
    lemma_to_bow = dict(zip(bol_df['lemma'], bol_df['bow_name']))
    replaced_lemmas = [[lemma_to_bow.get(lemma, lemma) for lemma in doc] for doc in tokenized_cleaned_tweets['lemmatized_tweets'].str.split(" ")]	

Here some random examples with the new mapping, you can inspect the differences in lemmas:

.. code-block:: python

    tweets['new_clean_lemmas'] = replaced_lemmas
    sample_tweets = tweets.sample(10, random_state=1)  # You can change the random_state for different samples
    print("Before and After Text Cleaning:")
    print('-' * 40)
    for index, row in sample_tweets.iterrows():
        print(f"Original: {row['Snippet']}")
        print(f"Cleaned: {row['new_clean_lemmas']}")
        print('-' * 40)


.. parsed-literal::

    Before and After Text Cleaning:
    ----------------------------------------
    Original: Antes el pasaporte venezolano permitía la entrada en en sinfín de países del mundo. Hoy cada día estamos más limitados gracias al socialismo del siglo 21. Hasta Cuba, que saquea a Venezuela, nos impone una visa. #PeroTodoTieneSuFinal
    Cleaned: ['pasaporte', 'venezolano', 'permitir', 'entrada', 'sinfin', 'pais', 'mundo', 'hoy', 'cada', 'diar', 'limitado', 'gracias', 'socialismo', 'siglo', 'cuba', 'saquear', 'venezuela', 'imponer', 'vis']
    ----------------------------------------
    Original: @VickyDavilaH Bueno y si @AlvaroUribeVel se proclama presidente de una vez por todas y nombra a @IvanDuque ministro de guerra y lo deja que solito libere al pueblo venezolano, ¿será que le prestan atención a la grave crisis que vive el Chocó, que parece que solo cuentan con el Esmad ?
    Cleaned: ['buen', 'proclamar', 'presidente', 'vez', 'todo', 'nombra', 'ministro', 'guerra', 'dejar', 'solitir', 'liderar', 'pueblo', 'venezolano', 'presentar', 'atencion', 'grave', 'crisis', 'vivir', 'choco', 'parecer', 'solo', 'contar', 'esmad']
    ----------------------------------------
    Original: @zonacero Nomás quieren Telesur y Venezolana de Televisión, super imparcialicimos.
    Cleaned: ['noma', 'querer', 'telesur', 'venezolano', 'television', 'super', 'imparcialicir']
    ----------------------------------------
    Original: RT @XiomaryUrbaez Sr @jguaido yo, venezolana y residente en el país, SÍ QUIERO INTERVENCIÓN. Le agradezco que sin haber hecho una consulta pública sobre algo tan importante, no hable por mí. Gracias.
    Cleaned: ['sr', 'venezolano', 'presidente', 'pai', 'querer', 'intervencion', 'agradecer', 'haber', 'hecho', 'consulta', 'publicar', 'tanto', 'importante', 'hablar', 'gracias']
    ----------------------------------------
    Original: Y también las grandes masas de venezolanos queriendo refugiarse en Colombia, de verdad que esto es una gran insensatez descarada y cruel, porque todo está premeditadamente calculado.
    Cleaned: ['grande', 'masa', 'venezolano', 'querer', 'refugiar', 'el', 'colombia', 'verdad', 'gran', 'insensatez', 'descarado', 'cruel', 'premeditadamente', 'calculado']
    ----------------------------------------
    Original: RT @fernandoperezm #Metro de Madrid #FelizViernesATodos talento venezolano https://t.co/Pe4wuvq6eU
    Cleaned: ['madrid', 'talento', 'venezolano']
    ----------------------------------------
    Original: Para que dejen de estar creyendo en los medios oficiales venezolanos. Hacen el mismo trabajo de varios medios colombianos de lavarle la cara al gobierno. https://t.co/msWCOzeCdH
    Cleaned: ['dejar', 'creer', 'medio', 'oficial', 'venezolano', 'hacer', 'mismo', 'trabajar', 'varios', 'medio', 'colombia', 'lavar', 'el', 'cara', 'gobierno']
    ----------------------------------------
    Original: RT @Crisantemonegro He visto hace dias venezolanos vendiendo cosas o trabajando, pero hoy por primera vez se me acercaron dos a pedirme dinero porque no tenian para comer; iban cargados con maletas y se les veia el recorrido que llevan. Fué inevitable no llorar frente a tan triste situación.
    Cleaned: ['visto', 'hacer', 'dia', 'venezolano', 'vender', 'cosa', 'trabajar', 'hoy', 'primero', 'vez', 'acercar', 'dos', 'pedir', 'yo', 'dinero', 'comer', 'ir', 'cargado', 'maleta', 'veiar', 'recorrido', 'llevar', 'inevitable', 'llorar', 'frente', 'tanto', 'triste', 'situacion']
    ----------------------------------------
    Original: El canciller uruguayo, Ernesto Talvi, aseguró que no “ha sido posible” enviar vuelos humanitarios al país para repatriar a los ciudadanos venezolanos Entérate24.com- E l canciller Jorge Arreaza, aseguró que Venezuela no ha recibido ninguna solicitud de vuelo por parte de Uruguay para repatriar a los ciudadanos venezolanos que se encuentran...
    Cleaned: ['canciller', 'uruguay', 'ernesto', 'talvi', 'asegurar', 'ser', 'posible', 'enviar', 'vuelo', 'humanitario', 'pai', 'repatriar', 'ciudadano', 'venezolano', 'enterate', 'com', 'l', 'canciller', 'jorge', 'arreazar', 'asegurar', 'venezuela', 'recibido', 'ningun', 'solicitud', 'vuelo', 'parte', 'uruguay', 'repatriar', 'ciudadano', 'venezolano', 'encontrar']
    ----------------------------------------
    Original: Si tan solo crearan la infraestructura necesaria para que llegara a cada hogar venezolano la mayoría no tendria falta de gas ... pero la 5ta no hace nada productivo ni constructivo😒 sino todo lo contrario...
    Cleaned: ['tanto', 'solo', 'crearar', 'infraestructura', 'necesario', 'llegar', 'cada', 'hogar', 'venezolano', 'mayorio', 'falta', 'gas', 'ta', 'hacer', 'producto', 'constructivo', 'sino', 'contrario']
    ----------------------------------------
    

From here, you can use this processed tweets to train different models and make your own empirical applications of NLP using social media data. However, we will show you a simple application of **Topic Modelling** using the data we processed. For more information about this methodology, we deliver some links to help understanding this type of unsupervised classification.

1. `Practical guide for Topic Modelling <https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/>`__
2. `An example of a fully developed real pipeline for Topic Modelling <https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24>`__
3. `Topic Modelling used in a Kaggle competition <https://www.kaggle.com/code/dikshabhati2002/topic-modeling-lda/notebook>`__
4. `Real Research derived from Topic Modelling <https://arxiv.org/abs/1711.04305>`__

Now we can plug this processed documents in a toy model to see some topics about Venezuelan migrants in Colombia:

This model resolves in some steps: 
1. We iterate over the best combination of hyperparameters ``alpha``, ``beta``, and number of topics. 
2. We filter the results and pick the model with best coherence. We calculate Coherence Score and Perplexity of each LDA Topic Modeling implementation. 
3. We display a visualization of the topics found in the toy model.

**NOTE:** This code takes a lot of time iterating over different combinations of hyperparameters, expect long kernel runs. You may adjust it for your use case.

.. code-block:: python

    # Now we create our initial variables for Topic Modeling
    # Create Dictionary
    dictionary = corpora.Dictionary(replaced_lemmas)
    corpus = [dictionary.doc2bow(text) for text in replaced_lemmas]
    # A function that resolves our hyperparameters using a corpus and a dictionary
    def compute_coherence_perplexity_values(corpus, dictionary, k, a, b):
        
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=k, 
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               alpha=a,
                                               eta=b,
                                               workers=7)
        
        coherence_model_lda = CoherenceModel(model=lda_model, texts=replaced_lemmas, dictionary=dictionary, coherence='c_v')
        
        return (coherence_model_lda.get_coherence(),lda_model.log_perplexity(corpus))

    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 2
    max_topics = 4
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
                   # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
                   gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
                   corpus]
    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': [],
                     'Perplexity': []
                    }

    # Can take a long time to run
    if 1 == 1:
        # This is the number of times we want to iterate to find optimal hyperparameters
        pbar = tqdm.tqdm(total = 540)
        
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        (cv, pp) = compute_coherence_perplexity_values(corpus=corpus_sets[i], dictionary=dictionary, 
                                                      k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        model_results['Perplexity'].append(pp)
                        pbar.update(1)
        pd.DataFrame(model_results).to_csv(os.path.join(data_folder,"lda_tuning_results.csv"), index=False)
        pbar.close()


Now we want to find the optimal model to train, let's see the results of our trainning pocess:

.. code-block:: python

    # Pre-trained models available in Github data folder, we recommend retraining the model with your own data
    tabla_tunning = pd.read_csv(os.path.join(data_folder,"lda_tuning_results.csv"))
    tabla_tunning = tabla_tunning.sort_values(by = 'Coherence', ascending = False)
    tabla_tunning


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
          <th>Validation_Set</th>
          <th>Topics</th>
          <th>Alpha</th>
          <th>Beta</th>
          <th>Coherence</th>
          <th>Perplexity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>117</th>
          <td>100% Corpus</td>
          <td>3</td>
          <td>asymmetric</td>
          <td>0.61</td>
          <td>0.417986</td>
          <td>-7.722477</td>
        </tr>
        <tr>
          <th>58</th>
          <td>75% Corpus</td>
          <td>3</td>
          <td>asymmetric</td>
          <td>0.9099999999999999</td>
          <td>0.406760</td>
          <td>-7.942932</td>
        </tr>
        <tr>
          <th>32</th>
          <td>75% Corpus</td>
          <td>3</td>
          <td>0.01</td>
          <td>0.61</td>
          <td>0.399778</td>
          <td>-7.888492</td>
        </tr>
        <tr>
          <th>43</th>
          <td>75% Corpus</td>
          <td>3</td>
          <td>0.61</td>
          <td>0.9099999999999999</td>
          <td>0.391599</td>
          <td>-8.025149</td>
        </tr>
        <tr>
          <th>39</th>
          <td>75% Corpus</td>
          <td>3</td>
          <td>0.31</td>
          <td>symmetric</td>
          <td>0.383571</td>
          <td>-7.980128</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>63</th>
          <td>100% Corpus</td>
          <td>2</td>
          <td>0.01</td>
          <td>0.9099999999999999</td>
          <td>0.292411</td>
          <td>-7.677486</td>
        </tr>
        <tr>
          <th>108</th>
          <td>100% Corpus</td>
          <td>3</td>
          <td>0.9099999999999999</td>
          <td>0.9099999999999999</td>
          <td>0.291343</td>
          <td>-7.848078</td>
        </tr>
        <tr>
          <th>104</th>
          <td>100% Corpus</td>
          <td>3</td>
          <td>0.61</td>
          <td>symmetric</td>
          <td>0.288476</td>
          <td>-7.852673</td>
        </tr>
        <tr>
          <th>102</th>
          <td>100% Corpus</td>
          <td>3</td>
          <td>0.61</td>
          <td>0.61</td>
          <td>0.281319</td>
          <td>-7.834575</td>
        </tr>
        <tr>
          <th>85</th>
          <td>100% Corpus</td>
          <td>2</td>
          <td>asymmetric</td>
          <td>0.01</td>
          <td>0.270411</td>
          <td>-8.548538</td>
        </tr>
      </tbody>
    </table>
    <p>120 rows × 6 columns</p>
    </div>


Let's train the model! We now pick the best result from the validation table created on the last step. We might want to revisit alternatives to the model that fits better with interpretability of the Topics found.

.. code-block:: python

    lda_final_model = gensim.models.LdaMulticore(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=3,
                                                 random_state=100,
                                                 chunksize=100,
                                                 passes=10,
                                                 alpha=0.01,
                                                 eta=0.61,
                                                 workers=7)

Now that we have trained an optimized version of our toy model, we want to visually inspect the derived topics and see if we find some interesting patterns giving information related to the way people speaks about Venezuelan migrants in Colombia.

.. code-block:: python

    [[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]
    
    pprint.pprint(lda_final_model.print_topics())
    doc_lda = lda_final_model[corpus]
    
    visxx = pyLDAvis.gensim_models.prepare(topic_model=lda_final_model, corpus=corpus, dictionary=dictionary)
    pyLDAvis.display(visxx)


.. parsed-literal::

    [(0,
      '0.049*"venezolano" + 0.015*"colombia" + 0.008*"el" + 0.005*"ver" + '
      '0.004*"hacer" + 0.004*"ir" + 0.004*"ayuda" + 0.004*"decir" + 0.004*"pai" + '
      '0.003*"ser"'),
     (1,
      '0.035*"venezolano" + 0.006*"pai" + 0.006*"colombia" + 0.005*"ser" + '
      '0.005*"decir" + 0.005*"poder" + 0.004*"hacer" + 0.003*"pueblo" + 0.003*"el" '
      '+ 0.003*"ir"'),
     (2,
      '0.039*"venezolano" + 0.007*"pai" + 0.007*"hacer" + 0.006*"colombia" + '
      '0.004*"migrant" + 0.004*"ser" + 0.004*"el" + 0.004*"poder" + 0.004*"ver" + '
      '0.004*"bogota"')]
    

.. raw:: html

    
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">
    
    
    <div id="ldavis_el1750018732114786401925045056" style="background-color:white;"></div>
    <script type="text/javascript">
    
    var ldavis_el1750018732114786401925045056_data = {"mdsDat": {"x": [-0.05921807152386709, 0.03357886840567159, 0.0256392031181955], "y": [-0.004412714385678792, -0.047162053182917614, 0.05157476756859637], "topics": [1, 2, 3], "cluster": [1, 1, 1], "Freq": [45.86867407124039, 30.235237627007983, 23.89608830175163]}, "tinfo": {"Term": ["pueblo", "pai", "bogota", "aqui", "hora", "parte", "venir", "salir", "migrant", "hablar", "criticar", "mal", "asi", "derecho", "venezolano", "deber", "pedir", "tratar", "poder", "primero", "igual", "hacer", "venecir", "allo", "ser", "tu", "yo", "presencia", "ano", "ud", "semaforo", "amor", "chino", "nestor", "delito", "nacido", "detalle", "uno", "moneda", "pedir", "dificil", "humberto", "lastimo", "sirio", "obrador", "general", "humano", "gota", "registrado", "diferente", "senalado", "arepa", "invasion", "caos", "proyecto", "comer", "dudamel", "brasileno", "santamario", "kobi", "asi", "correr", "video", "nuevo", "primero", "lado", "venecir", "vez", "colombia", "mes", "poner", "el", "venezolano", "ayuda", "tras", "tres", "via", "ver", "ir", "familia", "pais", "querer", "conocido", "vida", "hoy", "nino", "pasar", "solo", "dar", "frontera", "decir", "gobierno", "mismo", "hacer", "ano", "hora", "venezuela", "pai", "ser", "trabajar", "poder", "veneco", "buen", "analizar", "consulta", "patio", "leopoldir", "quitar", "fallecido", "laboral", "comercial", "refineria", "reunion", "show", "reporto", "empresario", "recordar", "acento", "sector", "impresionante", "recibido", "tampoco", "enfermedad", "reconstruir", "abril", "cualquiera", "investigar", "efecto", "exiliado", "dolor", "campo", "humillar", "vivar", "bogota", "detra", "regresar", "producto", "atender", "salir", "barrio", "venir", "veneca", "cuanto", "importante", "migrant", "parte", "mil", "venezolano", "ultimo", "hacer", "pai", "q", "noticia", "hora", "seguir", "cerca", "trabajar", "ser", "poder", "presidente", "ano", "ver", "colombia", "el", "aqui", "venezuela", "varios", "dar", "decir", "buen", "migracion", "llegar", "criticar", "fundacion", "programa", "hernandez", "zulia", "web", "causar", "mendigo", "ataque", "adorar", "despectivo", "izquierdo", "actividad", "acogida", "halla", "impotencia", "titular", "reforma", "inconciencia", "esmad", "muerto", "radical", "siguiente", "encuentro", "pj", "aristeguietar", "mentalidad", "arreglir", "alex", "reconciliacion", "queriar", "ojo", "semana", "tratar", "derecho", "tu", "yo", "ladron", "caer", "pueblo", "igual", "presencia", "ojalar", "mal", "deber", "venezolano", "aqui", "poder", "pai", "decir", "ser", "nunca", "allo", "hablar", "defender", "buen", "entrar", "colombia", "hacer", "hambre", "ir", "medio", "solo", "venezuela", "bien", "el", "querer", "veneco", "pasar", "presidente", "migrant", "senor"], "Freq": [19.0, 73.0, 20.0, 20.0, 30.0, 22.0, 23.0, 17.0, 35.0, 20.0, 5.0, 19.0, 26.0, 9.0, 579.0, 22.0, 22.0, 8.0, 48.0, 23.0, 12.0, 69.0, 24.0, 14.0, 54.0, 8.0, 8.0, 8.0, 26.0, 9.0, 7.0166792663345845, 6.6552267368083164, 6.133414312369838, 5.259332504342915, 5.809310192447676, 5.155421384005705, 5.0543380723408236, 5.139582762541767, 4.837636801725182, 19.224519512879596, 4.7400224374507935, 4.58018868852557, 4.631798728082101, 4.687337685953832, 4.404605998002231, 5.998936460669039, 4.241253760792895, 4.110371420809355, 4.282996845434701, 4.694551490967167, 6.171416878737146, 6.291339832859783, 4.100917788824613, 3.854737155218489, 3.8234283952876855, 4.012139422620345, 3.782939382317956, 3.743720351964239, 3.7283001160033553, 3.7281637881428096, 21.095002849792674, 6.4587395972391155, 10.560964253140684, 6.096419921268315, 18.419017850954436, 11.026457721842188, 18.75644702481095, 14.562207933675872, 91.18181264441981, 8.357878544181608, 14.811711784890525, 51.71661973771453, 303.577246254606, 25.931115455411224, 9.354158209701707, 12.334840677820438, 12.742545598145025, 32.903730808963466, 26.532277203709857, 10.697849719948946, 13.855343999010099, 20.96532142446397, 8.091867088632965, 12.518104376529061, 15.59081429792634, 13.762774127188266, 17.25365403067192, 20.594830755557116, 19.591253752568015, 14.621448717299344, 24.7181142689832, 15.250126273461767, 13.048421923726258, 26.86044765464485, 15.124050846608823, 16.24991357889511, 18.703488171143448, 24.70427753355635, 21.2530968816434, 15.122950098578315, 18.013920878985964, 14.88581772949892, 14.892187892871265, 4.875393039346119, 4.525010512157189, 3.8199377455123913, 3.5477137852490537, 5.908979876217818, 3.6320853070350765, 4.410997248507794, 3.230824483700008, 3.1807020394572825, 4.873174824961651, 3.2995197657257767, 2.8076415715270775, 2.7431181343494675, 5.054495854418032, 2.66081751806229, 5.248793890914946, 2.6143699299516667, 4.197259206386584, 2.6130741838315323, 2.587810948311197, 2.5800262219430103, 2.577975339011142, 6.789838862732992, 2.455850552096816, 2.413740948467488, 2.3288228582757498, 2.3879778638643856, 2.2592186947275255, 2.213041411271102, 2.2130406961794242, 14.58275589099695, 3.806811284311201, 7.276679474425636, 5.3893823449495075, 5.654099272779528, 11.244854771285889, 7.788508249706563, 14.135007434417304, 9.4984750380161, 4.587674472608683, 5.709901690218848, 18.29796416974865, 12.046497331920143, 9.325210231401568, 161.19606516415377, 7.037893201015209, 28.490772250864886, 28.86300511936001, 10.588285242961435, 6.6149093226925935, 13.489576174065924, 11.46538141706814, 5.800676857979375, 11.974943351730511, 17.526078957465675, 15.252588939760836, 10.167311725708947, 10.801686715266303, 14.876071241932465, 23.312608441691882, 17.132810952177024, 8.885072257184623, 11.953316119027784, 7.693476379552115, 9.792920233802503, 10.991964029622965, 9.382290756032969, 8.10666922262259, 8.022726040751557, 4.986051317301129, 3.7449564409174934, 4.020732025916612, 3.004681652710227, 2.9810830055436113, 3.4894521891589716, 2.6756097025843726, 2.5424010983587615, 2.473605652079184, 2.3168485119713145, 2.3354663612164024, 2.174451457432734, 2.0984548688803613, 2.0984544921037873, 2.098132724909492, 2.0889241170498067, 1.9873055911278148, 1.9873050259629537, 1.9868393301173461, 1.9845814964969553, 3.3390512698564416, 1.8476424274110206, 1.9672384707903896, 1.8082907516806093, 1.7949594545469452, 1.792574647221077, 1.7689309752538545, 1.8140706927158379, 1.7670442665589678, 1.7589284991525063, 2.728481817288654, 2.7568519629898764, 4.264402479035294, 5.3161087265565845, 6.17307046621447, 5.530384201692448, 5.565362631727094, 2.8605369200326085, 4.647157743829502, 10.577523058508175, 6.8875081776282965, 5.213330105562889, 3.772615985999933, 9.19807544733252, 9.718155223647985, 114.62175163572262, 9.239950395779788, 15.28001119579978, 20.421676136306335, 15.822373538708472, 15.943053568291573, 5.805887754005055, 6.333478571325524, 7.4050484798342175, 4.144859184376301, 9.742995349626312, 5.002234624712566, 20.137163102553465, 13.908291272159248, 5.902264187451369, 10.014269208117891, 7.337096071139224, 9.074692422606711, 9.104359056499467, 6.361259061688255, 10.466395822185214, 7.577662638840338, 6.6670757964948, 6.524797426577066, 6.331086793633035, 6.741462041969401, 5.774681987807035], "Total": [19.0, 73.0, 20.0, 20.0, 30.0, 22.0, 23.0, 17.0, 35.0, 20.0, 5.0, 19.0, 26.0, 9.0, 579.0, 22.0, 22.0, 8.0, 48.0, 23.0, 12.0, 69.0, 24.0, 14.0, 54.0, 8.0, 8.0, 8.0, 26.0, 9.0, 7.778716419806119, 7.40305388769803, 6.889371906658457, 6.008182295922825, 6.645279688097537, 5.90310989649639, 5.806033303093073, 5.909075050031394, 5.586756146804609, 22.217390401291752, 5.490269639656301, 5.328723809971771, 5.398151969750812, 5.467965324509209, 5.1551910275035056, 7.049258406960461, 4.989854025338762, 4.85805993330004, 5.062517149513182, 5.549700878421086, 7.326824243269533, 7.476024538430598, 4.8834579059558125, 4.603594933724369, 4.57351339057293, 4.802528816598182, 4.532889787988927, 4.493916160710127, 4.476329187419969, 4.476300994436195, 26.036364935730507, 7.826455532461113, 13.068940088900803, 7.44527961530351, 23.664298864228506, 13.83479607950207, 24.240800588694768, 18.729649071111968, 134.63158418866516, 10.464902944258169, 19.519173089304218, 79.31582651207678, 579.3950630544824, 37.08695571892843, 12.010488764968368, 16.63887429205564, 17.35505756502111, 52.95723589868963, 41.85316690635791, 14.281845180206325, 19.723750846970926, 33.22157481509793, 10.369519893065258, 18.031091255310074, 23.945055494939812, 20.86379071933392, 28.132041145864743, 35.689949138907814, 34.90190188616183, 23.601032797017734, 51.53245183731464, 25.854299175926307, 20.911936484015204, 69.25951117766898, 26.664594916610564, 30.174646020450666, 39.7611633466707, 73.98895878922269, 54.72222940740065, 28.959008466068802, 48.54652101454658, 28.899148083572683, 34.01747399853055, 5.680690543982476, 5.372642611749271, 4.657174694551471, 4.347075906585212, 7.281257426015624, 4.482935423755527, 5.4615788699202295, 4.0318563412322765, 3.9873817141188157, 6.12658737260398, 4.172124708257235, 3.613833103846169, 3.5456828685347506, 6.536875835210771, 3.460179639398448, 6.834046996868825, 3.413744414269163, 5.483473470615592, 3.4139147629970856, 3.387401608660259, 3.3793883432791683, 3.379652683496224, 8.940684291015774, 3.2585969583291683, 3.2142549228877906, 3.1281849796119077, 3.2178311395804347, 3.0588174553007645, 3.01240353260726, 3.012402817515582, 20.867427256580687, 5.284755818455752, 10.414152045317634, 7.597157646729014, 8.049702010377576, 17.354064705287545, 11.864113511031748, 23.845755908701307, 15.221465233249297, 6.5829172820191415, 8.49489221675271, 35.46409876554919, 22.393114234166493, 16.32678139280772, 579.3950630544824, 11.506961116766062, 69.25951117766898, 73.98895878922269, 19.809060276623665, 10.753687588185102, 30.174646020450666, 25.46349867189356, 9.128140167189258, 28.959008466068802, 54.72222940740065, 48.54652101454658, 23.72240895676313, 26.664594916610564, 52.95723589868963, 134.63158418866516, 79.31582651207678, 20.74450897472167, 39.7611633466707, 17.511892228324115, 34.90190188616183, 51.53245183731464, 34.01747399853055, 21.645675789931236, 25.579090190517434, 5.960658407995544, 4.598078489182628, 5.111775105419474, 3.845186519080212, 3.821485668689286, 4.4878566007617975, 3.526203822538294, 3.3819553284996475, 3.3332602302542504, 3.1564027421122, 3.189064924320435, 3.0344994884097862, 2.9380090990212473, 2.938008722244673, 2.938092412911484, 2.9310962084365793, 2.826859821268701, 2.8268592561038393, 2.8269106609277252, 2.8305123538311605, 4.854921405713805, 2.6871966575519064, 2.864570818762939, 2.6478449818214953, 2.634513684687831, 2.632128877361963, 2.6084852053947403, 2.6758206046840214, 2.6088056432128877, 2.6016222797137045, 4.064899478954148, 4.118953666222727, 6.482155135604558, 8.28544760444635, 9.874327296382608, 8.83441775304927, 8.996012543015667, 4.363307043469542, 7.529763238994711, 19.373511610775285, 12.412174043642743, 8.927461729636725, 6.202748395467898, 19.590321908606196, 22.037297851649235, 579.3950630544824, 20.74450897472167, 48.54652101454658, 73.98895878922269, 51.53245183731464, 54.72222940740065, 12.715999153835078, 14.920298626630359, 20.03939053942277, 7.772815004583671, 34.01747399853055, 10.767601141465764, 134.63158418866516, 69.25951117766898, 14.522212040163314, 41.85316690635791, 22.92533473516928, 35.689949138907814, 39.7611633466707, 17.70575287113075, 79.31582651207678, 33.22157481509793, 28.899148083572683, 28.132041145864743, 23.72240895676313, 35.46409876554919, 19.094946333992826], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -6.786, -6.8389, -6.9206, -7.0743, -6.9748, -7.0943, -7.1141, -7.0973, -7.1579, -5.7781, -7.1783, -7.2126, -7.2014, -7.1894, -7.2517, -6.9427, -7.2895, -7.3208, -7.2797, -7.1879, -6.9144, -6.8951, -7.3231, -7.385, -7.3932, -7.345, -7.4038, -7.4142, -7.4184, -7.4184, -5.6853, -6.8689, -6.3771, -6.9266, -5.8209, -6.334, -5.8028, -6.0559, -4.2215, -6.6111, -6.0389, -4.7885, -3.0187, -5.4789, -6.4985, -6.2219, -6.1894, -5.2407, -5.4559, -6.3643, -6.1056, -5.6914, -6.6435, -6.2071, -5.9876, -6.1123, -5.8863, -5.7093, -5.7592, -6.0518, -5.5268, -6.0097, -6.1656, -5.4437, -6.018, -5.9462, -5.8056, -5.5273, -5.6778, -6.0181, -5.8432, -6.0339, -6.0335, -6.7333, -6.8079, -6.9773, -7.0512, -6.5411, -7.0277, -6.8334, -7.1448, -7.1604, -6.7338, -7.1238, -7.2852, -7.3084, -6.6973, -7.3389, -6.6595, -7.3565, -6.8831, -7.357, -7.3667, -7.3697, -7.3705, -6.4021, -7.4191, -7.4364, -7.4722, -7.4471, -7.5025, -7.5232, -7.5232, -5.6377, -6.9807, -6.3329, -6.6331, -6.5852, -5.8976, -6.2649, -5.6689, -6.0664, -6.7942, -6.5753, -5.4107, -5.8288, -6.0848, -3.2349, -6.3662, -4.968, -4.955, -5.9578, -6.4282, -5.7156, -5.8782, -6.5596, -5.8347, -5.4538, -5.5928, -5.9984, -5.9378, -5.6178, -5.1685, -5.4765, -6.1332, -5.8365, -6.2772, -6.0359, -5.9204, -6.0787, -6.2248, -6.2353, -6.4756, -6.7618, -6.6908, -6.9821, -6.99, -6.8325, -7.0981, -7.1491, -7.1766, -7.242, -7.234, -7.3055, -7.341, -7.341, -7.3412, -7.3456, -7.3955, -7.3955, -7.3957, -7.3968, -6.8766, -7.4683, -7.4056, -7.4899, -7.4973, -7.4986, -7.5119, -7.4867, -7.5129, -7.5175, -7.0785, -7.0682, -6.6319, -6.4115, -6.262, -6.372, -6.3657, -7.0312, -6.546, -5.7235, -6.1525, -6.431, -6.7545, -5.8632, -5.8082, -3.3406, -5.8587, -5.3557, -5.0656, -5.3208, -5.3132, -6.3234, -6.2364, -6.0801, -6.6604, -5.8057, -6.4724, -5.0797, -5.4498, -6.3069, -5.7782, -6.0893, -5.8768, -5.8735, -6.232, -5.7341, -6.057, -6.1851, -6.2066, -6.2368, -6.174, -6.3288], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.6763, 0.6729, 0.6632, 0.6463, 0.6449, 0.644, 0.6407, 0.6399, 0.6354, 0.6347, 0.6325, 0.628, 0.6263, 0.6253, 0.622, 0.618, 0.6168, 0.6123, 0.6122, 0.612, 0.6078, 0.6069, 0.6047, 0.6019, 0.6003, 0.5996, 0.5985, 0.5967, 0.5965, 0.5965, 0.5689, 0.5873, 0.5663, 0.5795, 0.5288, 0.5525, 0.5229, 0.5277, 0.3897, 0.5546, 0.5034, 0.3517, 0.133, 0.4216, 0.5294, 0.4801, 0.4705, 0.3035, 0.3236, 0.4904, 0.4262, 0.3191, 0.5314, 0.4145, 0.3503, 0.3633, 0.2905, 0.2296, 0.2019, 0.3006, 0.0447, 0.2515, 0.3077, -0.1678, 0.2123, 0.1605, 0.0252, -0.3176, -0.1664, 0.1297, -0.212, 0.116, -0.0466, 1.0433, 1.0245, 0.998, 0.993, 0.9873, 0.9857, 0.9825, 0.9747, 0.9701, 0.9673, 0.9615, 0.9437, 0.9395, 0.939, 0.9335, 0.9322, 0.9294, 0.9289, 0.9288, 0.9269, 0.9263, 0.9254, 0.921, 0.9133, 0.9097, 0.9011, 0.8979, 0.8932, 0.8878, 0.8878, 0.8378, 0.8681, 0.8377, 0.8528, 0.8429, 0.7622, 0.7753, 0.6732, 0.7246, 0.8351, 0.7989, 0.5344, 0.5762, 0.6361, -0.0832, 0.7045, 0.3079, 0.2548, 0.5698, 0.7102, 0.3911, 0.3982, 0.7428, 0.3131, 0.0576, 0.0384, 0.3489, 0.2925, -0.0736, -0.5574, -0.3363, 0.3483, -0.0057, 0.3737, -0.0747, -0.3489, -0.0919, 0.214, 0.0367, 1.2529, 1.2262, 1.1914, 1.1848, 1.1831, 1.1798, 1.1554, 1.1461, 1.1332, 1.1222, 1.1199, 1.0982, 1.0949, 1.0949, 1.0947, 1.0927, 1.0791, 1.0791, 1.0788, 1.0764, 1.0571, 1.0569, 1.0557, 1.0501, 1.0477, 1.0473, 1.0431, 1.0428, 1.0419, 1.04, 1.0328, 1.0299, 1.0127, 0.9877, 0.9617, 0.9631, 0.9512, 1.0092, 0.9488, 0.8263, 0.8425, 0.8935, 0.9342, 0.6754, 0.6127, -0.1889, 0.6227, 0.2755, 0.1441, 0.2507, 0.1982, 0.6475, 0.5746, 0.4359, 0.8027, 0.1811, 0.6648, -0.4685, -0.1739, 0.5311, 0.0013, 0.2922, 0.0621, -0.0427, 0.4078, -0.5938, -0.0465, -0.0352, -0.0298, 0.1105, -0.2288, 0.2355]}, "token.table": {"Topic": [2, 2, 3, 3, 3, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 3, 3, 3, 1, 2, 3, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 3, 2, 1, 3, 1, 2, 1, 1, 2, 3, 1, 2, 1, 2, 3, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 3, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 3, 2, 3, 2, 1, 3, 3, 2, 2, 1, 2, 3, 1, 2, 3, 3, 1, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 3, 1, 2, 3, 3, 2, 3, 1, 2, 1, 2, 3, 3, 1, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 2, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3, 3, 1, 2, 3, 2, 1, 2, 2, 3, 1, 1, 2, 3, 2, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 1, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 2, 3, 3], "Freq": [0.8876651777414369, 0.8670070090700694, 0.6807331730696758, 0.6807330857709969, 0.6336327026067785, 0.7666343428853097, 0.5361822977002131, 0.06702278721252664, 0.4021367232751598, 0.9455557268916006, 0.8801746832163692, 0.562543704373166, 0.4125320498736551, 0.03750291362487774, 0.14461658281021092, 0.4338497484306328, 0.4338497484306328, 0.8025655840422841, 0.13376093067371403, 0.7598412134000404, 0.7474342624086988, 0.8065642055577832, 0.15363127724910156, 0.03840781931227539, 0.6000131588428205, 0.7453692064954547, 0.24845640216515155, 0.701055114823839, 0.16178194957473205, 0.1348182913122767, 0.25286339322448953, 0.6743023819319721, 0.08428779774149651, 0.2823940917052167, 0.33887291004626, 0.33887291004626, 0.1916863037698417, 0.7188236391369064, 0.09584315188492085, 0.8900922618387082, 0.4409498483234809, 0.26456990899408855, 0.29396656554898726, 0.26561260115623736, 0.6640315028905933, 0.6538474522348853, 0.868886176474251, 0.8507732822547073, 0.32865402426480944, 0.6573080485296189, 0.8709066778933947, 0.6759186601598456, 0.1708365844360049, 0.14855355168348253, 0.8328945338496389, 0.7440741301519426, 0.7714918417148798, 0.19287296042871996, 0.09643648021435998, 0.9306407221402835, 0.7666305615759674, 0.12777176026266124, 0.12777176026266124, 0.16776670152052567, 0.8388335076026283, 0.111848262107283, 0.782937834750981, 0.111848262107283, 0.15190833442969764, 0.7595416721484882, 0.15190833442969764, 0.5730346748791288, 0.2865173374395644, 0.17191040246373868, 0.4083985278315987, 0.18151045681404385, 0.45377614203510963, 0.48513119614265093, 0.2134577263027664, 0.3104839655312966, 0.25730703726006465, 0.12865351863003233, 0.5146140745201293, 0.9028965343244608, 0.30381816502062164, 0.10127272167354055, 0.6076363300412433, 0.6271430803266523, 0.8611731519583825, 0.18922350139768768, 0.7568940055907507, 0.9009494582746813, 0.9107020835342811, 0.6215366541144155, 0.882439279816386, 0.6222281828857356, 0.6556068604048699, 0.21433301205543823, 0.1260782423855519, 0.8460993583556858, 0.7553312273682153, 0.8856345797115331, 0.46435598183007765, 0.46435598183007765, 0.7065858579606467, 0.6393483803020262, 0.892272500470026, 0.7702086012839054, 0.21005689125924693, 0.07001896375308231, 0.6355654063535484, 0.2118551354511828, 0.16948410836094624, 0.8699286037440077, 0.8511533630368239, 0.14185889383947065, 0.5801743028473554, 0.2707480079954325, 0.11603486056947107, 0.8233739506961647, 0.5988206066643011, 0.0499017172220251, 0.3493120205541757, 0.3898381542245924, 0.4042766043810587, 0.20213830219052936, 0.6807137825927377, 0.2754401319122329, 0.34430016489029114, 0.41316019786834934, 0.7801962232816771, 0.5302464853823341, 0.43082526937314647, 0.6681964050315606, 0.2923359272013078, 0.08352455062894508, 0.8016266567494305, 0.9383109686869825, 0.6639216752839828, 0.241698189974748, 0.241698189974748, 0.563962443274412, 0.23543559458657, 0.70630678375971, 0.117717797293285, 0.6823385715703895, 0.8788004126671739, 0.7074861005135716, 0.8190917331593344, 0.6137610835509684, 0.6451124728604093, 0.11946527275192764, 0.23893054550385529, 0.6590872753938376, 0.8935949581969104, 0.18309723686451237, 0.7323889474580495, 0.7950966488257702, 0.14456302705923096, 0.2291839629981292, 0.2291839629981292, 0.6875518889943877, 0.926242912022132, 0.9201587655602148, 0.5082276149454018, 0.3127554553510165, 0.15637772767550825, 0.25522806737562886, 0.25522806737562886, 0.4594105212761319, 0.3925787825550607, 0.26171918837004043, 0.3053390530983805, 0.8870608002178739, 0.7667285196265246, 0.7644600282116711, 0.09555750352645889, 0.09555750352645889, 0.4619860380913415, 0.36958883047307317, 0.18479441523653659, 0.2819753031399258, 0.5075555456518664, 0.19738271219794803, 0.3062452959774792, 0.5512415327594625, 0.12249811839099166, 0.6216545277830688, 0.19127831624094424, 0.19127831624094424, 0.8949737322721327, 0.20597655789506503, 0.617929673685195, 0.8470111666000996, 0.8321984510012319, 0.6710190007334847, 0.28757957174292204, 0.09585985724764068, 0.0929913568531304, 0.6509394979719129, 0.2789740705593912, 0.8058797399183251, 0.13431328998638753, 0.23592326200298747, 0.31456434933731664, 0.47184652400597493, 0.7759169308488404, 0.3224377118796759, 0.16121885593983795, 0.6448754237593518, 0.24278010413189394, 0.7283403123956819, 0.3378882526407646, 0.39195037306328695, 0.2703106021126117, 0.7098041396193183, 0.10140059137418833, 0.20280118274837666, 0.4465658458858934, 0.5358790150630721, 0.6042931585324696, 0.14218662553705166, 0.24882659468984042, 0.8588898339329393, 0.8551859447406259, 0.0900195731305922, 0.0450097865652961, 0.7591533919995501, 0.3707783714224639, 0.3089819761853866, 0.3089819761853866, 0.7684751772716972, 0.15369503545433943, 0.051231678484779816, 0.33604176538117464, 0.11201392179372488, 0.5600696089686243, 0.2950796444306445, 0.42154234918663497, 0.252925409511981, 0.7606394807331144, 0.042257748929617464, 0.16903099571846986, 0.13162817549673383, 0.6581408774836691, 0.13162817549673383, 0.19562675966315612, 0.7825070386526245, 0.8746011344899364, 0.15485060531470404, 0.3097012106294081, 0.5677855528205815, 0.2524097524151822, 0.5553014553134008, 0.20192780193214577, 0.632119341629052, 0.1505046051497743, 0.24080736823963886, 0.24600854342830847, 0.7380256302849254, 0.8240334943470415, 0.13733891572450693, 0.7442700534697906, 0.1823661599456479, 0.7294646397825916, 0.7687511041072764, 0.8877346120833716, 0.15297827665832614, 0.7648913832916306, 0.7523734157122149, 0.7074989657449482, 0.7901207802100275, 0.1920463606923457, 0.67216226242321, 0.09602318034617285, 0.830143483053251, 0.16322300477940796, 0.8161150238970398, 0.17287016332755412, 0.633857265534365, 0.17287016332755412, 0.89358933012375, 0.7316309065903212, 0.14632618131806424, 0.4319909114508968, 0.4319909114508968, 0.11781570312297186, 0.8998913988144167, 0.3085393604689577, 0.6170787209379154, 0.8189086841426063, 0.13648478069043438, 0.5236987748008487, 0.1571096324402546, 0.3142192648805092, 0.38375629478940704, 0.3289339669623489, 0.29238574841097675, 0.23968602808560732, 0.719058084256822, 0.3490924341789702, 0.6981848683579404, 0.9144169180423227, 0.588400950594424, 0.16811455731269256, 0.25217183596903886, 0.8787565619729449, 0.7074988242969882, 0.517973535508837, 0.4143788284070696, 0.06906313806784493, 0.7493450246796599, 0.08326055829773998, 0.08326055829773998, 0.24138707954977695, 0.6034676988744424, 0.7212026360298601, 0.18030065900746503, 0.06010021966915501, 0.22638730201655724, 0.11319365100827862, 0.6791619060496717, 0.10131003745994814, 0.6078602247596888, 0.3039301123798444, 0.3476156701504669, 0.6083274227633171, 0.8461561171021912, 0.39972836223135544, 0.45683241397869195, 0.11420810349467299, 0.26278679080529804, 0.5912702793119206, 0.13139339540264902, 0.7838024957336215, 0.0412527629333485, 0.20626381466674248, 0.5190464423595428, 0.24222167310111997, 0.24222167310111997, 0.5246851749087373, 0.2778760301325879, 0.19848287866613418, 0.4778532216057737, 0.3018020346983834, 0.22635152602378755, 0.3354894695152358, 0.5871065716516626, 0.08387236737880895, 0.6231443057777973, 0.2832474117171806, 0.09441580390572687, 0.8008692497680342, 0.16017384995360684, 0.053391283317868944, 0.7490611858413728, 0.23048036487426854, 0.057620091218567135, 0.7209768846448248, 0.11091952071458842, 0.16637928107188266, 0.8416902920338648, 0.15303459855161178, 0.6639218328873624, 0.22282351887764276, 0.6684705566329282, 0.222320721590452, 0.111160360795226, 0.666962164771356, 0.78503499949771], "Term": ["abril", "acento", "acogida", "actividad", "adorar", "alex", "allo", "allo", "allo", "amor", "analizar", "ano", "ano", "ano", "aqui", "aqui", "aqui", "arepa", "arepa", "aristeguietar", "arreglir", "asi", "asi", "asi", "ataque", "atender", "atender", "ayuda", "ayuda", "ayuda", "barrio", "barrio", "barrio", "bien", "bien", "bien", "bogota", "bogota", "bogota", "brasileno", "buen", "buen", "buen", "caer", "caer", "campo", "caos", "causar", "cerca", "cerca", "chino", "colombia", "colombia", "colombia", "comer", "comercial", "conocido", "conocido", "conocido", "consulta", "correr", "correr", "correr", "criticar", "criticar", "cualquiera", "cualquiera", "cualquiera", "cuanto", "cuanto", "cuanto", "dar", "dar", "dar", "deber", "deber", "deber", "decir", "decir", "decir", "defender", "defender", "defender", "delito", "derecho", "derecho", "derecho", "despectivo", "detalle", "detra", "detra", "diferente", "dificil", "dolor", "dudamel", "efecto", "el", "el", "el", "empresario", "encuentro", "enfermedad", "entrar", "entrar", "esmad", "exiliado", "fallecido", "familia", "familia", "familia", "frontera", "frontera", "frontera", "fundacion", "general", "general", "gobierno", "gobierno", "gobierno", "gota", "hablar", "hablar", "hablar", "hacer", "hacer", "hacer", "halla", "hambre", "hambre", "hambre", "hernandez", "hora", "hora", "hoy", "hoy", "hoy", "humano", "humberto", "humillar", "igual", "igual", "igual", "importante", "importante", "importante", "impotencia", "impresionante", "inconciencia", "invasion", "investigar", "ir", "ir", "ir", "izquierdo", "kobi", "laboral", "laboral", "lado", "lado", "ladron", "ladron", "ladron", "lastimo", "leopoldir", "llegar", "llegar", "llegar", "mal", "mal", "mal", "medio", "medio", "medio", "mendigo", "mentalidad", "mes", "mes", "mes", "migracion", "migracion", "migracion", "migrant", "migrant", "migrant", "mil", "mil", "mil", "mismo", "mismo", "mismo", "moneda", "muerto", "muerto", "nacido", "nestor", "nino", "nino", "nino", "noticia", "noticia", "noticia", "nuevo", "nuevo", "nunca", "nunca", "nunca", "obrador", "ojalar", "ojalar", "ojalar", "ojo", "ojo", "pai", "pai", "pai", "pais", "pais", "pais", "parte", "parte", "pasar", "pasar", "pasar", "patio", "pedir", "pedir", "pedir", "pj", "poder", "poder", "poder", "poner", "poner", "poner", "presencia", "presencia", "presencia", "presidente", "presidente", "presidente", "primero", "primero", "primero", "producto", "producto", "producto", "programa", "programa", "proyecto", "pueblo", "pueblo", "pueblo", "q", "q", "q", "querer", "querer", "querer", "queriar", "queriar", "quitar", "quitar", "radical", "recibido", "recibido", "reconciliacion", "reconstruir", "recordar", "recordar", "refineria", "reforma", "registrado", "regresar", "regresar", "regresar", "reporto", "reunion", "reunion", "salir", "salir", "salir", "santamario", "sector", "sector", "seguir", "seguir", "seguir", "semaforo", "semana", "semana", "senalado", "senalado", "senor", "senor", "senor", "ser", "ser", "ser", "show", "show", "siguiente", "siguiente", "sirio", "solo", "solo", "solo", "tampoco", "titular", "trabajar", "trabajar", "trabajar", "tras", "tras", "tras", "tratar", "tratar", "tres", "tres", "tres", "tu", "tu", "tu", "ud", "ud", "ud", "ultimo", "ultimo", "uno", "varios", "varios", "varios", "veneca", "veneca", "veneca", "venecir", "venecir", "venecir", "veneco", "veneco", "veneco", "venezolano", "venezolano", "venezolano", "venezuela", "venezuela", "venezuela", "venir", "venir", "venir", "ver", "ver", "ver", "vez", "vez", "vez", "via", "via", "via", "vida", "vida", "vida", "video", "video", "vivar", "web", "web", "yo", "yo", "yo", "zulia"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 3, 2]};
    
    function LDAvis_load_lib(url, callback){
      var s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onreadystatechange = s.onload = callback;
      s.onerror = function(){console.warn("failed to load library " + url);};
      document.getElementsByTagName("head")[0].appendChild(s);
    }
    
    if(typeof(LDAvis) !== "undefined"){
       // already loaded: just create the visualization
       !function(LDAvis){
           new LDAvis("#" + "ldavis_el1750018732114786401925045056", ldavis_el1750018732114786401925045056_data);
       }(LDAvis);
    }else if(typeof define === "function" && define.amd){
       // require.js is available: use it to load d3/LDAvis
       require.config({paths: {d3: "https://d3js.org/d3.v5"}});
       require(["d3"], function(d3){
          window.d3 = d3;
          LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
            new LDAvis("#" + "ldavis_el1750018732114786401925045056", ldavis_el1750018732114786401925045056_data);
          });
        });
    }else{
        // require.js not available: dynamically load d3 & LDAvis
        LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
             LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                     new LDAvis("#" + "ldavis_el1750018732114786401925045056", ldavis_el1750018732114786401925045056_data);
                })
             });
    }
    </script>


.. code-block:: python

    # We save the model in order to be able to use it later
    pyLDAvis.save_html(visxx, "lda_final_model.html")

Conclusion and Analysis of toy-Topic Modelling Results
------------------------------------------------------

In the realm of unsupervised classification, Topic Modelling has emerged as a powerful tool. It meticulously unravels the underlying topics within a corpus of text, illuminating the subtle narratives interwoven within large textual data. This technique finds its prowess particularly accentuated when applied to a heterogeneous assortment of tweets. By incorporating a more substantial and varied datas enhanced. Here, we simply made a basic aproximation for Topic Modelling to show how **tidyX** can be useful in order to process social media data preparing it to NLP tasks like this one.

For those eager to delve deeper into this subject, we recommend reaching out to `Barómetro de Xenofobia <https://barometrodexenofobia.org/nosotres/>`__, a reservoir of comprehensive data that can greatly augment research in this field.

Given λ = 0.5, and navigating through a dataset comprising 1000 tweets, our toy exploration has yielded the following intriguing topics:

**Topic 1: Migrant Necessities and Frontier Struggles**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Some Interesting Relevant Words:** ayuda, pedir, querer, primero, niño, frontera, gobierno, vida.

This topic unveils the urgent necessities and pleas echoed by the migrants. It portrays a vivid picture of their journey, marked by vulnerability and struggle, especially among children at the frontiers. The narrative fluctuates between government interventions and intrinsic human endeavors for survival.

**Topic 2: Migrant Flows and Economic Perceived Competition for Resources**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Some Interesting Relevant Words:** Bogotá, migrante, venir, salir, regresar, cualquiera, trabajar, quitar, sector, mil

Focusing on the economic dynamics within principal cities such as Bogotá, this topic elucidates the Perceived Competition for Resources and flow of the migrant populace. It highlights the intricate tapestry of employment, competition, and the transformative economic landscapes molded by the presence of migrants. There is a vast interesting literature that studies this labour market externalities and impact evaluation when migrants start seeking jobs in a foreign country.

**Topic 3: Advocacy and Critique—The Landscape of Migrant Rights and Initiatives**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Some Interesting Relevant Words:** mal, poder, decir, ser, deber, criticar, derecho, igual, tratar, programa, fundación, presencia

This topic blossoms into a vibrant discourse revolving around rights, responsibilities, and critiques of migrants in Colombia. It encapsulates a conversation of political and social agendas related to migrants context in Colombia.

**In closing, our analysis, though constrained by the volume of the dataset, serves as a gateway to exploring the vast universe of modelling tasks that** tidyX*\* attemps to address. It invites further exploration, promising a richer and evolving way to analyze social media data.*\*
