from setuptools import find_packages, setup

setup(
    name = 'tidytweets',
    packages = find_packages(include = ['tidytweets']),
    version = '0.1',
    description = 'Clean tweets to perform various NLP tasks such as topic analysis, word embeddings, sentiment analysis, etc.',
    author = 'Lucas Gómez Tobón, Jose Fernando Barrera',
    author_email  = "lucasgomeztobon@hotmail.com, jf.barrera10@uniandes.edu.co",
    license = 'MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/lgomezt/Tidytweets',
    keywords='NLP Text processing Twitter API processing data cleaning',
    install_requires = [],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    # Running `python setup.py pytest`
    # will execute all tests stored in the "tests" folder.
    test_suite = 'tests'
)