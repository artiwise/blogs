from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
#!pip install TurkishStemmer
from TurkishStemmer import TurkishStemmer
stemmer = TurkishStemmer()
tokenizer = RegexpTokenizer(r'\w+')
punct_re=lambda x :" ".join(tokenizer.tokenize(x.lower()))

def stemmer_char(text,i):
    return " ".join([word[:i] for word in word_tokenize(text)])
  
data["cleaned_text"]=data["text"].apply(punct_re)
data["stemmer4"]=data["cleaned_text"].apply(lambda x : stemmer_char(x.lower(),4))
data["stemmer5"]=data["cleaned_text"].apply(lambda x : stemmer_char(x,5))
data["TurkishStemmer"]=data["cleaned_text"].apply(lambda x : " ".join([stemmer.stem(w) for w in x.split()]))