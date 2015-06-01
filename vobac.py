import nltk, re, pprint,urllib2,numpy
from nltk import word_tokenize ,FreqDist

url = "http://www.gutenberg.org/files/2554/2554.txt"
response = urllib2.urlopen(url)
raw = response.read()
tokens=word_tokenize(raw)
text = nltk.Text(tokens)
words = [w.lower() for w in tokens]
vocab = sorted(set(words))

fre=FreqDist(set(words))
print(fre)
print(type(words))
print(text[1:10])
print(text[5])
print(text[9])
print(vocab[:10])
print(numpy.asarray(vocab).shape)