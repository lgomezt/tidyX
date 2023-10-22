Remove Unwanted Elements
-------------------------

Special characters 
^^^^^^^^^^^^^^^^^^

Using `remove_special_characters() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_special_characters>`_ function, the user can eliminate all characters that aren't lowercase letters or spaces. However, if desired, numbers can be preserved.   
   
.. code-block:: python
      
      from tidyX import TextPreprocessor as tp

      # Example string with special characters
      string_example = "This is an example text! It contains special characters. 123"
      print("Before:", string_example)

      # Remove special characters, excluding numbers
      cleaned_text = tp.remove_special_characters(string=string_example)
      print("After:", cleaned_text)

      # Remove special characters, but retain numbers
      cleaned_text_with_numbers = tp.remove_special_characters(string=string_example, allow_numbers=True)
      print("After (with numbers):", cleaned_text_with_numbers)

.. parsed-literal::

      Before: This is an example text! It contains special characters. 123
      After: his is an example text it contains special characters 
      After (with numbers): his is an example text it contains special characters 123

Note that the uppercase characters "T" and "I" were removed. To run this function ensure to lowercase the string beforehand (for example using :code:`.lower()` method).

Emojis and/or accents
^^^^^^^^^^^^^^^^^^^^^^

With the `remove_accents(string, delete_emojis=True) <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_accents>`_ function, you can eliminate accents from characters. Optionally, you can also remove emojis.

.. code-block:: python

      from tidyX import TextPreprocessor as tp

      # Example text with accents and emojis
      sample_text = "Café ☕️ à côté de l'hôtel. 😃"
      print("Original:", sample_text)

      # Use remove_accents function without deleting emojis
      without_accents = tp.remove_accents(string=sample_text, delete_emojis=False)
      print("Without Accents:", without_accents)

      # Use remove_accents function and delete emojis
      without_accents_or_emojis = tp.remove_accents(string=sample_text, delete_emojis=True)
      print("Without Accents or Emojis:", without_accents_or_emojis)

.. parsed-literal::

      Original: Café ☕️ à côté de l'hôtel. 😃
      Without Accents: Cafe ☕️ a cote de l'hotel. 😃
      Without Accents or Emojis: Cafe a cote de l'hotel. 

URLs
^^^^

To remove URLs from a text, you can utilize the `remove_urls() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_urls>`_ function.

.. code-block:: python
    
    from tidyX import TextPreprocessor as tp

    # Sample text containing URLs
    text_with_urls = "Check out our website: http://example.com. For more info, visit http://example2.com"
    print("Original:", text_with_urls)
    
    # Removing URLs from the text
    cleaned_text = tp.remove_urls(text_with_urls)
    print("Cleaned:", cleaned_text)


.. parsed-literal::

    Original: Check out our website: http://example.com. For more info, visit http://example2.com
    Cleaned: Check out our website:  For more info, visit 

Extra spaces
^^^^^^^^^^^^

To eliminate extra spaces in a text, you can use the `remove_extra_spaces() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_extra_spaces>`_ function.

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    # Sample text with redundant spaces
    text_with_spaces = "This is    an   example  text with extra   spaces.     "
    print("Original:", text_with_spaces)
    
    # Removing unnecessary spaces from the text
    refined_text = tp.remove_extra_spaces(text_with_spaces)
    print("Refined:", refined_text)


.. parsed-literal::

    Original: This is    an   example  text with extra   spaces.     
    Refined: This is an example text with extra spaces.


Mentions, RT prefix or hashtags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To remove mentions from tweets, use the `remove_mentions() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_mentions>`_ function. To eliminate the "RT" prefix, apply the `remove_RT() <some_link>`_ function.

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    # Sample tweet containing mentions
    tweet_sample = "Exploring the beauty of nature with @NatureExplorer and @WildlifeEnthusiast. #NaturePhotography 🌼"
    print("Original:", tweet_sample)
    
    # Removing mentions from the tweet and extracting them
    refined_text, mentions_list = tp.remove_mentions(tweet_sample, extract=True)
    print("Refined:", refined_text)
    print("Extracted Mentions:", mentions_list)


.. parsed-literal::

    Original: Exploring the beauty of nature with @NatureExplorer and @WildlifeEnthusiast. #NaturePhotography 🌼
    Refined: Exploring the beauty of nature with  and . #NaturePhotography 🌼
    Extracted Mentions: ['@WildlifeEnthusiast', '@NatureExplorer']

.. code-block:: python

    from tidyX import TextPreprocessor as tp
    
    # Sample tweet with the "RT" prefix
    tweet_sample = "RT     @username: Check out this amazing article!"
    print("Original:", tweet_sample)
    
    # Removing the "RT" prefix from the tweet
    refined_tweet = tp.remove_RT(tweet_sample)
    print("Refined:", refined_tweet)

.. parsed-literal::

    Original: RT     @username: Check out this amazing article!
    Refined: @username: Check out this amazing article!

Stopwords or any other concrete word 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With `remove_words() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_words>`_, users can effortlessly exclude specific words or even general stopwords from a given text.

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    # Original text
    text = "She was not only beautiful but also extremely talented in the field of music."
    print("Original:", text)
    
    # Apply remove_words function to remove English stopwords
    refined_text = tp.remove_words(string=text, remove_stopwords=True, language="english")
    print("Refined:", refined_text)

.. parsed-literal::

    Original: She was not only beautiful but also extremely talented in the field of music.
    Refined: She beautiful extremely talented field music.

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    # Original text
    text = "I love spending my weekends hiking in the mountains or swimming in the lake."
    print("Original:", text)
    
    # Define the bag of words that we want to remove
    words_to_remove = ["hiking", "swimming"]
    
    # Apply remove_words function
    refined_text = tp.remove_words(string=text, bag_of_words=words_to_remove)
    print("Refined:", refined_text)

.. parsed-literal::

    Original: I love spending my weekends hiking in the mountains or swimming in the lake.
    Refined: I love spending my weekends in the mountains or in the lake.

Repetition of characters 
^^^^^^^^^^^^^^^^^^^^^^^^

The `remove_repetitions() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_repetitions>`_ function is adept at removing redundant consecutive characters from a string. Consider the example 'coooroosooo', which would be transformed to 'coroso'. However, several languages, including English and Spanish, often feature double letters like 'll'. To accommodate such cases, the function provides an exceptions argument that specifies characters allowed to repeat once.

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    string_example = "Goooal ⚽️⚽️⚽️ Christiano Ronaldo Amazing Goal Juventus vs Real Madrid 1-3 Champions League Final #JUVRMA #UCLFinal2017 #JuventusRealMadrid"
    print("Before:", string_example)
    string_without_repetitions = tp.remove_repetitions(string = string_example, exceptions = None)
    print("After:", string_without_repetitions)

.. parsed-literal::

    Before: Goooal ⚽️⚽️⚽️ Christiano Ronaldo Amazing Goal Juventus vs Real Madrid 1-3 Champions League Final #JUVRMA #UCLFinal2017 #JuventusRealMadrid
    After: Goal ⚽️⚽️⚽️ Christiano Ronaldo Amazing Goal Juventus vs Real Madrid 1-3 Champions League Final #JUVRMA #UCLFinal2017 #JuventusRealMadrid
    
However, it's worth noting that there exist numerous words that feature the repetition of a single character. To address this, the :code:`remove_repetitions` function incorporates the :code:`exceptions` parameter, which allows for specifying a list of characters that are permitted to appear twice. For instance, if we set :code:`exceptions = ['p']`, words such as 'happpy' will be cleaned and transformed into 'happy'. The default value for this parameter is :code:`['r', 'l', 'n', 'c', 'a', 'e', 'o']`. Let’s see another example:

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    string_example = "HAPPPYYYYY GRADUATION TO US!! THANKYOUUUU LORD!!! 🫶🤍"
    print("Before:", string_example)
    string_without_repetitions = tp.remove_repetitions(string = string_example,exceptions = ["P"])
    print("After:", string_without_repetitions)

.. parsed-literal::

    Before: HAPPPYYYYY GRADUATION TO US!! THANKYOUUUU LORD!!! 🫶🤍
    After: HAPPY GRADUATION TO US! THANKYOU LORD! 🫶🤍
    

The `remove_last_repetition() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.remove_last_repetition>_` function focuses on removing repeated characters at the end of words. This is especially handy when working with social media texts where users tend to stress words by repeating their ending characters. The function helps standardize such text, ensuring a cleaner representation.

.. code-block:: python

    from tidyX import TextPreprocessor as tp

    string_example = "Here's Johnnyyy!"
    print("Before:", string_example)

    # Apply the remove_last_repetition function to standardize the text
    refined_text = tp.remove_last_repetition(string=string_example)
    print("After:", refined_text)

.. parsed-literal::

    Before: Here's Johnnyyy!
    After: Here's Johnny!