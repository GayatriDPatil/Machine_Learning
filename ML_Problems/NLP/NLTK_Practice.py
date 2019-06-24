import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#nltk.download()

ps = PorterStemmer()
example_words = ["python","pythonly","pythoner","pythoning","pythoned"]

for w in example_words:
    print(ps.stem(w))


