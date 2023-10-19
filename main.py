from tidyX import TextPreprocessor

string_example = "Holaaaaa! Cómo vasss?"
string_without_repetitions = TextPreprocessor.TextPreprocessor.remove_repetitions(string=string_example, exceptions=None)
print("After:", string_without_repetitions)
