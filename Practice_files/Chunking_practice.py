import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import ne_chunk

with open (r"C:\Flexon_Resume_Parser\Parser_Build-Arnav\MLK.txt", "r") as file:
    text = file.read()

tokens = word_tokenize(text)

tags = nltk.pos_tag(tokens)
ne_NER = ne_chunk(tags)

#print(ne_NER)

sentence = sent_tokenize(text)

sample = sentence[6]
sample_tokens = nltk.pos_tag(word_tokenize(sample))

grammar_np = r"NP: {<DT>?<JJ>*<NN>}"

chunk_parser = nltk.RegexpParser(grammar_np)
chunk_result = chunk_parser.parse(sample_tokens)

print(chunk_result)