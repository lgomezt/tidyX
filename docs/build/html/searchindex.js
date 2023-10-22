Search.setIndex({"docnames": ["index", "tutorials/stemming_and_lemmatizing_efficiently", "tutorials/topic_modelling", "tutorials/word_cloud", "usage/dependency_parsing_visualization", "usage/group_similar_terms", "usage/remove_unwanted_elements", "usage/standardize_text_pipeline", "usage/stemming_and_lemmatizing", "user_documentation/TextNormalization", "user_documentation/TextPreprocessor", "user_documentation/TextVisualizer"], "filenames": ["index.rst", "tutorials\\stemming_and_lemmatizing_efficiently.rst", "tutorials\\topic_modelling.rst", "tutorials\\word_cloud.rst", "usage\\dependency_parsing_visualization.rst", "usage\\group_similar_terms.rst", "usage\\remove_unwanted_elements.rst", "usage\\standardize_text_pipeline.rst", "usage\\stemming_and_lemmatizing.rst", "user_documentation\\TextNormalization.rst", "user_documentation\\TextPreprocessor.rst", "user_documentation\\TextVisualizer.rst"], "titles": ["Welcome to tidyX\u2019s documentation!", "Stemming and Lemmatizing Texts Efficiently", "Topic Modelling", "Word Cloud", "Dependency Parsing Visualization", "Group similar terms", "Remove Unwanted Elements", "Standardize Text Pipeline", "Stemming and Lemmatizing", "TextNormalization", "TextPreprocessor", "TextVisualizer"], "terms": {"python": [0, 4, 8, 9, 11], "packag": [0, 3], "design": 0, "clean": [0, 6, 7, 10], "preprocess": [0, 7, 10], "text": [0, 4, 5, 6, 8, 10, 11], "machin": 0, "learn": 0, "applic": [0, 8], "especi": [0, 6], "written": 0, "spanish": [0, 4, 6, 8, 9, 10], "origin": [0, 6, 8, 10], "from": [0, 3, 4, 5, 6, 7, 8, 10], "social": [0, 5, 6, 10], "network": [0, 5], "thi": [0, 3, 5, 6, 7, 8, 9, 10], "librari": [0, 8, 10], "provid": [0, 6, 7, 8, 10, 11], "complet": 0, "pipelin": 0, "remov": [0, 7, 9, 10], "unwant": 0, "charact": [0, 4, 7, 10], "normal": [0, 8], "group": [0, 10], "similar": [0, 10], "term": 0, "etc": [0, 10], "facilit": [0, 4], "nlp": [0, 5, 8], "us": [0, 3, 6, 9, 10, 11], "pip": [0, 9, 10], "make": [0, 4, 8], "sure": [0, 4, 8], "you": [0, 3, 4, 5, 6, 8, 9], "have": [0, 4, 8, 10], "necessari": [0, 4, 8], "depend": [0, 8, 10, 11], "If": [0, 8, 10, 11], "plan": [0, 8], "lemmat": [0, 9], "ll": [0, 4, 6, 8], "need": [0, 3, 4, 8, 9], "spaci": [0, 4, 8, 9, 11], "along": [0, 4, 8], "appropri": [0, 4, 8], "languag": [0, 4, 6, 7, 8, 9, 10, 11], "model": [0, 3, 4, 8, 9, 11], "For": [0, 4, 6, 7, 8, 9, 10, 11], "we": [0, 4, 5, 6, 8], "recommend": [0, 4, 8], "download": [0, 4, 8, 9, 10, 11], "es_core_web_sm": [0, 4, 8], "m": [0, 4, 8, 9, 11], "english": [0, 4, 6, 7, 8], "suggest": [0, 4, 8], "en_core_web_sm": [0, 4, 8], "To": [0, 5, 6, 7, 8, 10, 11], "see": [0, 6], "full": 0, "list": [0, 6, 10], "avail": [0, 9], "differ": 0, "visit": [0, 6, 7, 9], "standard": [0, 6], "stem": [0, 9], "element": 0, "special": [0, 7, 10], "emoji": [0, 7, 9, 10], "accent": [0, 7, 9, 10], "url": [0, 7, 10], "extra": [0, 7, 10], "space": [0, 7, 10], "hashtag": [0, 7, 10], "stopword": [0, 7, 10], "ani": [0, 10], "other": [0, 10], "concret": 0, "word": [0, 5, 8, 9, 10], "repetit": [0, 10], "pars": [0, 11], "visual": [0, 11], "effici": [0, 8], "cloud": 0, "topic": [0, 3], "textpreprocessor": [0, 3, 5, 6, 7], "textnorm": [0, 3, 8], "textvisu": [0, 3, 4], "In": [3, 5, 10], "tutori": [3, 8], "below": 3, "find": 3, "exampl": [3, 6, 7, 9, 10], "each": [3, 5, 10], "function": [3, 4, 5, 6, 7, 8, 9, 10], "within": [3, 8, 10], "our": [3, 6], "addition": [3, 7], "s": [3, 4, 5, 6, 8, 9, 10, 11], "util": [3, 4, 6, 10], "first": 3, "import": [3, 4, 5, 6, 7, 8, 10], "modul": 3, "tidyx": [3, 4, 5, 6, 7, 8, 9, 10, 11], "tp": [3, 5, 6, 7], "tn": [3, 8], "tv": [3, 4], "incorpor": 6, "tool": 4, "enabl": [], "displai": [4, 11], "linguist": 4, "analysi": 9, "featur": 6, "engin": [], "dependency_parse_visualizer_text": [4, 11], "when": [5, 6], "work": [5, 6], "corpu": 5, "sourc": [5, 9, 10, 11], "common": [5, 8, 10], "encount": [5, 10], "grammat": 5, "error": 5, "aren": [5, 6], "t": [5, 6, 9], "formal": 5, "includ": [5, 6, 9], "dictionari": [5, 9], "These": 5, "irregular": 5, "can": [4, 5, 6, 10], "pose": 5, "challeng": 5, "creat": 5, "frequenc": 5, "matric": 5, "algorithm": 5, "address": [5, 6], "develop": 5, "create_bol": [5, 10], "which": [4, 5, 6, 9, 10], "allow": [5, 6, 10], "specif": [5, 6], "bag": [5, 6, 10], "cluster": [5, 10], "relat": 5, "numpi": 5, "np": [5, 10], "arrai": [5, 10], "appl": [5, 10], "apl": 5, "banana": [5, 10], "banan": 5, "cherri": 5, "cheri": 5, "appli": [5, 6, 7, 8, 10], "bol_df": 5, "lemma": [5, 10], "print": [5, 6, 7, 8, 10], "bow_id": [5, 10], "bow_nam": [5, 10], "threshold": [5, 10], "1": [5, 6, 10], "100": 5, "86": 5, "89": 5, "91": 5, "2": [5, 10], "85": 5, "92": 5, "3": [5, 6], "4": 5, "5": [5, 10], "note": [5, 6, 8, 9, 10, 11], "datafram": [5, 10], "where": [5, 6, 10], "row": [5, 10], "correspond": [5, 10], "case": [5, 6], "all": [5, 6, 7, 8, 10], "three": 5, "categori": 5, "remove_special_charact": [6, 10], "user": [4, 6, 7], "elimin": [6, 10], "lowercas": [6, 7, 10], "letter": [6, 10], "howev": [6, 8, 10], "desir": 6, "number": [6, 10], "preserv": [6, 10], "string": [6, 7, 9, 10, 11], "string_exampl": 6, "an": [6, 7, 9, 10, 11], "It": [6, 7, 10], "contain": [6, 10], "123": 6, "befor": [6, 9, 10], "exclud": 6, "cleaned_text": [6, 7], "after": [6, 10], "retain": [6, 10], "cleaned_text_with_numb": 6, "allow_numb": [6, 10], "true": [4, 6, 7, 10, 11], "hi": 6, "uppercas": [6, 10], "i": 6, "were": 6, "run": [6, 8, 9], "ensur": [6, 10, 11], "beforehand": 6, "lower": 6, "method": [6, 7, 10], "With": 6, "remove_acc": [6, 10], "delete_emoji": [6, 10], "option": [6, 7, 9, 10, 11], "also": [6, 10], "sample_text": 6, "caf\u00e9": 6, "\u00e0": 6, "c\u00f4t\u00e9": 6, "de": [6, 10], "l": [6, 10], "h\u00f4tel": 6, "without": [6, 10], "delet": [6, 7], "without_acc": 6, "fals": [6, 10], "without_accents_or_emoji": 6, "cafe": 6, "cote": 6, "hotel": 6, "remove_url": [6, 10], "sampl": 6, "text_with_url": 6, "check": [6, 7, 9], "out": [6, 7], "websit": 6, "http": [6, 7, 9, 10, 11], "com": [6, 7], "more": [6, 7, 9], "info": 6, "example2": 6, "remove_extra_spac": [6, 10], "redund": 6, "text_with_spac": 6, "unnecessari": 6, "refined_text": 6, "refin": 6, "mention": [6, 7, 10], "tweet": [6, 7, 10], "remove_ment": [6, 10], "rt": [6, 7, 10], "prefix": [6, 7, 10], "remove_rt": [6, 10], "tweet_sampl": 6, "explor": 6, "beauti": 6, "natur": 6, "natureexplor": 6, "wildlifeenthusiast": 6, "naturephotographi": 6, "extract": [6, 7, 10], "them": 6, "mentions_list": 6, "usernam": [6, 10], "amaz": 6, "articl": 6, "refined_tweet": 6, "remove_word": [6, 10], "effortlessli": 6, "even": 6, "gener": 6, "given": [6, 9, 10, 11], "she": 6, "wa": 6, "onli": 6, "extrem": 6, "talent": 6, "field": 6, "music": 6, "remove_stopword": [6, 7, 10], "love": 6, "spend": 6, "my": 6, "weekend": 6, "hike": 6, "mountain": 6, "swim": 6, "lake": 6, "defin": [6, 10], "want": 6, "words_to_remov": 6, "bag_of_word": [6, 10], "The": [4, 6, 7, 9, 10, 11], "remove_repetit": [6, 10], "adept": 6, "consecut": [6, 10], "consid": [6, 9], "coooroosooo": [6, 10], "would": [6, 9, 10], "transform": [6, 7, 10], "coroso": [6, 10], "sever": 6, "often": [6, 9], "doubl": 6, "like": 6, "accommod": [6, 10], "except": [6, 10], "argument": 6, "specifi": [4, 6, 7, 10], "repeat": [6, 10], "onc": [6, 10], "goooal": 6, "christiano": 6, "ronaldo": 6, "goal": 6, "juventu": 6, "vs": 6, "real": 6, "madrid": 6, "champion": 6, "leagu": 6, "final": 6, "juvrma": 6, "uclfinal2017": 6, "juventusrealmadrid": 6, "string_without_repetit": 6, "none": [6, 10, 11], "worth": 6, "exist": 6, "numer": 6, "singl": [6, 8, 10], "paramet": [6, 7, 10], "ar": [4, 6, 8, 9, 10], "permit": 6, "appear": [6, 10], "twice": 6, "instanc": [6, 9], "set": [6, 10], "p": 6, "happpi": 6, "happi": 6, "default": [6, 9, 10, 11], "valu": 6, "r": [6, 10], "n": [6, 10], "c": [6, 10], "e": [6, 10], "o": [6, 10], "let": 6, "anoth": 6, "happpyyyyi": 6, "graduat": 6, "TO": 6, "thankyouuuu": 6, "lord": 6, "thankyou": 6, "remove_last_repetit": [6, 10], "some_link": 6, "_": 6, "focus": 6, "end": [6, 10], "handi": 6, "media": [6, 10], "tend": 6, "stress": 6, "help": 6, "cleaner": 6, "represent": [6, 11], "here": 6, "johnnyyi": 6, "johnni": 6, "encompass": 7, "solut": 7, "quickli": 7, "effect": 7, "input": [7, 8, 10], "particular": 7, "focu": 7, "strip": [7, 9], "certain": 7, "offer": 7, "trim": [7, 10], "retweet": [7, 10], "raw": [7, 10], "raw_tweet": 7, "link": 7, "simpli": 7, "add": [7, 10], "language_stopword": [7, 10], "addit": [7, 11], "detail": 7, "explan": 7, "customiz": 7, "step": [7, 8, 10], "offici": 7, "document": [4, 7, 10, 11], "One": 8, "foundat": 8, "prepar": 8, "bring": 8, "base": [8, 9, 10], "root": [8, 9], "both": [8, 10], "stemmer": [8, 9], "perform": [8, 10], "task": 8, "across": 8, "variou": 8, "token": [8, 9, 10], "load": [4, 8, 10], "model_en": 8, "en_core_news_sm": 8, "lemmatized_token": 8, "corriendo": 8, "model_": 8, "es_core_news_sm": [8, 9, 11], "correr": 8, "built": 8, "receiv": 8, "most": [4, 8, 10], "requir": 8, "do": 8, "refer": [8, 11], "class": [9, 10, 11], "text_norm": 9, "static": [9, 10, 11], "is_emoji": 9, "str": [9, 10, 11], "bool": [9, 10, 11], "process": [9, 10], "reduc": [9, 10], "its": [9, 10], "form": 9, "take": 9, "account": [9, 10], "mean": 9, "sentenc": 9, "leverag": 9, "vocabulari": 9, "morpholog": 9, "should": [9, 10], "name_of_model": 9, "es_core_news_md": 9, "es_core_news_lg": 9, "es_dep_news_trf": 9, "inform": [9, 10, 11], "io": [9, 11], "arg": [9, 10, 11], "A": [9, 10, 11], "object": [9, 11], "return": [9, 10, 11], "version": [9, 10], "snowbal": 9, "suffix": 9, "might": 9, "unlik": 9, "doesn": 9, "alwai": 9, "produc": 9, "valid": [9, 10], "context": 9, "support": 9, "multipl": 9, "instal": [4, 8, 9, 10], "nltk": [9, 10], "done": 9, "alreadi": 9, "text_preprocessor": 10, "ndarrai": 10, "verbos": 10, "levenshtein": 10, "distanc": 10, "handl": 10, "misspel": 10, "data": 10, "aim": 10, "togeth": 10, "possibl": 10, "same": 10, "progress": 10, "everi": 10, "increment": 10, "pd": 10, "column": 10, "id": 10, "int": 10, "repres": 10, "name": [4, 10], "score": 10, "fuzz": 10, "ratio": 10, "determin": [4, 10], "between": 10, "length": 10, "being": 10, "compar": 10, "sensit": 10, "toward": 10, "shorter": 10, "get_most_common_str": 10, "union": 10, "num_str": 10, "tupl": 10, "retriev": 10, "serv": 10, "primarili": 10, "descript": 10, "about": 10, "collect": 10, "flat": 10, "occurr": 10, "count": 10, "orang": 10, "rais": [10, 11], "valueerror": [10, 11], "non": 10, "posit": 10, "empti": [10, 11], "load_stopword": 10, "cach": 10, "must": 10, "dataset": 10, "typic": 10, "seri": 10, "follow": 10, "convert": 10, "entir": 10, "g": 10, "elonmusk": 10, "while": 10, "indic": 10, "separ": 10, "usual": 10, "begin": 10, "vari": 10, "white": 10, "mark": 10, "potenti": 10, "surround": 10, "lead": 10, "trail": 10, "replac": 10, "mai": [4, 10], "remove_hashtag": 10, "scan": 10, "start": 10, "alphanumer": 10, "last": 10, "emphas": 10, "holaaaa": 10, "amigooo": 10, "hola": 10, "amigo": 10, "duplic": 10, "uniqu": 10, "some": 10, "As": 10, "result": 10, "punctuat": 10, "exclam": 10, "whether": [10, 11], "sequenc": 10, "continu": 10, "until": 10, "line": 10, "predefin": 10, "space_between_emoji": 10, "insert": 10, "around": 10, "unnest_token": 10, "df": 10, "input_column": 10, "id_col": 10, "unnest": 10, "flatten": 10, "panda": 10, "split": 10, "turn": 10, "expect": 10, "identifi": 10, "ad": 10, "index": 10, "dedupl": 10, "concaten": 10, "thei": [4, 10], "text_visu": 11, "style": [4, 11], "dep": [4, 11], "jupyt": [4, 11], "entiti": [4, 11], "displaci": [4, 11], "usag": 11, "relev": 11, "sh": 11, "ent": [4, 11], "intend": 11, "notebook": [4, 11], "render": 11, "html": 11, "environ": 11, "otherwis": [10, 11], "directli": 11, "invalid": 11, "recognit": 4, "ner": 4, "individu": 4, "By": 4, "graphic": 4, "showcas": 4, "attribut": 4, "oper": 4, "choos": 4, "forc": 4, "luke": 4, "skywalk": 4, "darth": 4, "vader": 4, "icon": 4, "star": 4, "war": 4, "univers": 4, "show": 4, "as_datafram": 10, "either": 10, "entri": 10}, "objects": {"tidyX": [[9, 0, 0, "-", "text_normalization"], [10, 0, 0, "-", "text_preprocessor"], [11, 0, 0, "-", "text_visualizer"]], "tidyX.text_normalization": [[9, 1, 1, "", "TextNormalization"]], "tidyX.text_normalization.TextNormalization": [[9, 2, 1, "", "is_emoji"], [9, 2, 1, "", "lemmatizer"], [9, 2, 1, "", "stemmer"]], "tidyX.text_preprocessor": [[10, 1, 1, "", "TextPreprocessor"]], "tidyX.text_preprocessor.TextPreprocessor": [[10, 2, 1, "", "create_bol"], [10, 2, 1, "", "get_most_common_strings"], [10, 2, 1, "", "load_stopwords"], [10, 2, 1, "", "preprocess"], [10, 2, 1, "", "remove_RT"], [10, 2, 1, "", "remove_accents"], [10, 2, 1, "", "remove_extra_spaces"], [10, 2, 1, "", "remove_hashtags"], [10, 2, 1, "", "remove_last_repetition"], [10, 2, 1, "", "remove_mentions"], [10, 2, 1, "", "remove_repetitions"], [10, 2, 1, "", "remove_special_characters"], [10, 2, 1, "", "remove_urls"], [10, 2, 1, "", "remove_words"], [10, 2, 1, "", "space_between_emojis"], [10, 2, 1, "", "unnest_tokens"]], "tidyX.text_visualizer": [[11, 1, 1, "", "TextVisualizer"]], "tidyX.text_visualizer.TextVisualizer": [[11, 2, 1, "", "dependency_parse_visualizer_text"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"welcom": 0, "tidyx": 0, "s": 0, "document": 0, "instal": 0, "usag": 0, "tutori": 0, "user": 0, "stem": [1, 8], "lemmat": [1, 8], "text": [1, 7], "effici": 1, "topic": 2, "model": 2, "word": [3, 6], "cloud": 3, "depend": 4, "pars": 4, "visual": 4, "group": 5, "similar": 5, "term": 5, "remov": 6, "unwant": 6, "element": 6, "special": 6, "charact": 6, "emoji": 6, "accent": 6, "url": 6, "extra": 6, "space": 6, "hashtag": 6, "stopword": 6, "ani": 6, "other": 6, "concret": 6, "repetit": 6, "standard": 7, "pipelin": 7, "textnorm": 9, "textpreprocessor": 10, "textvisu": 11}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})