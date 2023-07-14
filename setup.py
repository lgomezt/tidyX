from setuptools import find_packages, setup

setup(
    name = 'tidytweets',
    packages = find_packages('tidytweets',include = ['tidytweets']),
    version = '0.15',
    description = 'Clean tweets to perform various NLP tasks such as topic analysis, word embeddings, sentiment analysis, etc.',
    author = 'Lucas Gómez Tobón, Jose Fernando Barrera',
    author_email  = "lucasgomeztobon@hotmail.com, jf.barrera10@uniandes.edu.co",
    license = 'MIT',
    package_dir={'': 'tidytweets'},
    url='https://github.com/lgomezt/Tidytweets',
    keywords='NLP Text processing Twitter API processing data cleaning',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    install_requires = [
        'pandas','numpy','nltk','spacy','gensim','textblob','tweepy',
        'emoji','unidecode','scikit-learn','matplotlib','seaborn','wordcloud',
        'plotly','pyLDAvis','thefuzz','tqdm','spacy_spanish_lemmatizer'
    ],
    # Running `python setup.py pytest`
    # will execute all tests stored in the "tests" folder.
    test_suite = 'tests',
)