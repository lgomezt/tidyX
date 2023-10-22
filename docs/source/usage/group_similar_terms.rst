Group similar terms
--------------------

When working with a corpus sourced from social networks, it's common to encounter texts with grammatical errors or words that aren't formally included in dictionaries. These irregularities can pose challenges when creating Term Frequency matrices for NLP algorithms. To address this, we developed the `create_bol() <https://tidyx.readthedocs.io/en/latest/user_documentation/TextPreprocessor.html#tidyX.text_preprocessor.TextPreprocessor.create_bol>`_ function, which allows you to create specific bags of terms to cluster related terms.

.. code-block:: python

   from tidyX import TextPreprocessor as tp
   import numpy as np
   
   # Create a numpy array of words to cluster
   words = np.array(['apple', 'aple', 'apples', 'banana', 'banan', 'bananas', 'cherry', 'cheri', 'cherries'])

   # Apply create_bol function to group similar words
   bol_df = tp.create_bol(lemmas = lemmas)
   
   print(bol_df)

+--------+---------+---------+-----------+----------+
| bow_id | bow_name|  lemma  | similarity| threshold|
+========+=========+=========+===========+==========+
|   1    |  apple  |  apple  |    100    |    86    |
+--------+---------+---------+-----------+----------+
|   1    |  apple  |   aple  |     89    |    86    |
+--------+---------+---------+-----------+----------+
|   1    |  apple  |  apples |     91    |    86    |
+--------+---------+---------+-----------+----------+
|   2    |  banana |  banana |    100    |    85    |
+--------+---------+---------+-----------+----------+
|   2    |  banana |  banan  |     91    |    85    |
+--------+---------+---------+-----------+----------+
|   2    |  banana | bananas |     92    |    85    |
+--------+---------+---------+-----------+----------+
|   3    |  cherry |  cherry |    100    |    85    |
+--------+---------+---------+-----------+----------+
|   4    |   cheri |   cheri |    100    |    86    |
+--------+---------+---------+-----------+----------+
|   5    |cherries | cherries|    100    |    85    |
+--------+---------+---------+-----------+----------+

Note that :code:`bol_df` is a dataframe where each row corresponds to a word from the :code:`words` array. In this case, the function groups all the words into three categories: apple, banana, and cherry. 
