from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

#nltk.download()


#First Program nltk - word and sentence
from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "Hello mr. smith, how are you doing today? The weather is great and python is awesome."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))
for i in word_tokenize(example_text):
    print(i)


# try stop words
example_sentence = "This is an example showing of stopword and filtering."
stop_words = set(stopwords.words("english"))
print(stop_words)
