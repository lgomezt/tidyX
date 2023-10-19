from tidyX import text_preprocessor

text_preprocessor.preprocess("Hola! ¿Cómo estás?")
string_example = "Holaaaaa! Cómo vasss?"
string_without_repetitions = TextPreprocessor.TextPreprocessor.remove_repetitions(string=string_example, exceptions=None)
print("After:", string_without_repetitions)
