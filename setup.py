from setuptools import find_packages, setup

setup(
    name = 'tidytweets',
    packages = find_packages(include = ['tidytweets']),
    version = '1.0',
    description = 'Clean tweets to perform topic analysis',
    author = 'Lucas Gómez Tobón, Jose Fernando Barrera',
    author_email  = "lucasgomeztobon@hotmail.com, jf.barrera10@uniandes.edu.co",
    license = 'MIT',
    install_requires = [],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    # Running `python setup.py pytest`
    # will execute all tests stored in the "tests" folder.
    test_suite = 'tests'
)