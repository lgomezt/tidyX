from setuptools import setup, find_packages
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding = 'utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name = "tidyX",
    version = "1.6.2",
    description = "Python package to clean raw tweets for ML applications",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "",
    author = "Lucas Gómez Tobón, Jose Fernando Barrera",
    author_email = "lucasgomeztobon@gmail.com, jf.barrera10@uniandes.edu.co",
    license = "MIT",
    classifiers = [
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages = ["tidyX"],
    include_package_data = True,
    install_requires = [
        "numpy",
        "emoji",
        "pandas",
        "regex",
        "spacy",
        "thefuzz",
        "Unidecode",
        "nltk"
        ]
)