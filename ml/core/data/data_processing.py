import re
import string

import contractions
import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

PUNCTUATION_LIST = list(string.punctuation)
stopword = nltk.corpus.stopwords.words('english')
tknzr = TweetTokenizer()
ps = PorterStemmer()
wn = WordNetLemmatizer()
tok = Tokenizer()


class DataProcessing:

    def __init__(self, data_column):
        self.data_column = data_column

    def process(self, data):
        self.clean(data)
        self.tokenization(data)
        self.clean_stopwords(data)
        self.stemlem(data)
        data = data[self.data_column]
        sequences = tok.texts_to_sequences(data)
        data = sequence.pad_sequences(sequences, maxlen=32)

        return data

    def clean(self, data):
        data[self.data_column] = data[self.data_column].str.lower()
        data[self.data_column] = data[self.data_column].apply(self.remove_urls)
        data[self.data_column] = data[self.data_column].apply(self.remove_placeholders)
        data[self.data_column] = data[self.data_column].apply(self.remove_html_references)
        data[self.data_column] = data[self.data_column].apply(self.remove_non_letter_characters)
        data[self.data_column] = data[self.data_column].apply(self.remove_mentions)
        data[self.data_column] = data[self.data_column].apply(self.remove_digits)
        data[self.data_column] = data[self.data_column].apply(self.expand_contractions)
        data[self.data_column] = data[self.data_column].apply(self.remove_punctuation)

    @staticmethod
    def remove_urls(text):
        text = re.sub(r'https?:\/\/\S+', '', text)
        return re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', text)

    @staticmethod
    def remove_placeholders(text):
        text = re.sub(r'{link}', '', text)
        return re.sub(r"\[video\]", '', text)

    @staticmethod
    def remove_html_references(text):
        return re.sub(r'&[a-z]+;', '', text)

    @staticmethod
    def remove_non_letter_characters(text):
        return re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', text)

    @staticmethod
    def remove_mentions(text):
        return re.sub(r'@mention', '', text)

    @staticmethod
    def remove_digits(text):
        return re.sub('[0-9]+', '', text)

    @staticmethod
    def expand_contractions(text):
        return contractions.fix(text)

    @staticmethod
    def remove_punctuation(text):
        return "".join([char for char in text if char not in PUNCTUATION_LIST])

    def tokenization(self, data):
        data[self.data_column] = data[self.data_column].apply(tknzr.tokenize)

    def clean_stopwords(self, data):
        data[self.data_column] = data[self.data_column].apply(self.remove_stopwords)

    @staticmethod
    def remove_stopwords(text):
        text = [word for word in text if word not in stopword]
        return text

    def stemlem(self, data):
        data[self.data_column] = data[self.data_column].apply(self.stemming)
        data[self.data_column] = data[self.data_column].apply(self.lemmatization)

    @staticmethod
    def stemming(text):
        return [ps.stem(word) for word in text]

    @staticmethod
    def lemmatization(text):
        return [wn.lemmatize(word) for word in text]
