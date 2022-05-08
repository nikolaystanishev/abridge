import re
import string
from copy import deepcopy

import contractions
import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud

le = LabelEncoder()
PUNCTUATION_LIST = list(string.punctuation)
stopword = nltk.corpus.stopwords.words('english')
tknzr = TweetTokenizer()
ps = PorterStemmer()
wn = WordNetLemmatizer()
tok = Tokenizer()


class DataProcessing:

    def __init__(self, dataset, is_runtime=False):
        self.dataset = dataset
        self.is_runtime = is_runtime
        self.PROCEDURES = {
            'drop_unused_columns': self.drop_unused_columns,
            'encode_labels': self.encode_labels,
            'lowercase': self.to_lowercase,
            'remove_urls': self.remove_urls,
            'remove_urls_1': self.remove_urls_1,
            'remove_placeholders': self.remove_placeholders,
            'remove_placeholders_1': self.remove_placeholders_1,
            'remove_frames': self.remove_frames,
            'remove_html_references': self.remove_html_references,
            'remove_non_letter_characters': self.remove_non_letter_characters,
            'remove_mentions': self.remove_mentions,
            'remove_mentions_1': self.remove_mentions_1,
            'remove_digits': self.remove_digits,
            'expand_contractions': self.expand_contractions,
            'remove_punctuation': self.remove_punctuation,
            'tokenize': self.tokenize,
            'remove_stopwords': self.remove_stopwords,
            'stem': self.stem,
            'lemmatize': self.lemmatize,
            'split': self.split,
            'to_list': self.to_list,
            'text_to_sequences': self.text_to_sequences,
            'create_embedding_matrix': self.create_embedding_matrix,
            'visualize': self.visualize
        }

    def proceed(self):
        steps = self.dataset.processing
        if self.is_runtime:
            steps = self.dataset.runtime_processing

        for procedure in steps:
            self.PROCEDURES.get(procedure)()

    def drop_unused_columns(self):
        columns = deepcopy(self.dataset.columns)
        columns.remove(self.dataset.data_column)
        columns.remove(self.dataset.label_column)
        self.dataset.dataset_df = self.dataset.dataset_df.drop(columns, axis=1)

    def encode_labels(self):
        self.dataset.dataset_df[self.dataset.label_column] = le.fit_transform(
            self.dataset.dataset_df[self.dataset.label_column])

    def to_lowercase(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[
            self.dataset.data_column].str.lower()

    def remove_urls(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'https?:\/\/\S+', self.dataset.replace_character, text))
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", self.dataset.replace_character, text))

    def remove_urls_1(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'http?:\/\/\S+', self.dataset.replace_character, text))
        self.remove_urls()

    def remove_placeholders(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'{link}', self.dataset.replace_character, text))
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r"\[video\]", self.dataset.replace_character, text))

    def remove_placeholders_1(self):
        self.remove_placeholders()
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'<\w+>', self.dataset.replace_character, text))

    def remove_frames(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'"\d+x\d+ custom picture frame / poster frame (\d+|\d+\.\d+) "',
                                self.dataset.replace_character, text))
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'frame \( \d+ this frame is', self.dataset.replace_character, text))

    def remove_html_references(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'&[a-z]+;', self.dataset.replace_character, text))

    def remove_non_letter_characters(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", self.dataset.replace_character, text))

    def remove_mentions(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'@mention', self.dataset.replace_character, text))

    def remove_mentions_1(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'@\w+', self.dataset.replace_character, text))

    def remove_digits(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: re.sub(r'\d+', self.dataset.replace_character, text))

    def expand_contractions(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: contractions.fix(text))

    def remove_punctuation(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: "".join(
                [char if char not in PUNCTUATION_LIST else self.dataset.replace_character for char in text]))

    def tokenize(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            tknzr.tokenize)

    def remove_stopwords(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: [word for word in text if word not in stopword])

    def stem(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: [ps.stem(word) for word in text])

    def lemmatize(self):
        self.dataset.dataset_df[self.dataset.data_column] = self.dataset.dataset_df[self.dataset.data_column].apply(
            lambda text: [wn.lemmatize(word) for word in text])

    def split(self):
        self.dataset.X_train, self.dataset.X_test, self.dataset.Y_train, self.dataset.Y_test = \
            train_test_split(self.dataset.dataset_df[self.dataset.data_column],
                             self.dataset.dataset_df[self.dataset.label_column], test_size=self.dataset.test_ratio)

    def to_list(self):
        self.dataset.X_train = self.dataset.X_train[self.dataset.data_column]

    def text_to_sequences(self):
        self.dataset.tokenizer = Tokenizer()
        self.dataset.tokenizer.fit_on_texts(self.dataset.X_train)

        sequences = self.dataset.tokenizer.texts_to_sequences(self.dataset.X_train)
        self.dataset.X_train = sequence.pad_sequences(sequences, maxlen=self.dataset.max_length,
                                                      padding=self.dataset.sequence_padding)

        if self.dataset.X_test is not None:
            sequences = self.dataset.tokenizer.texts_to_sequences(self.dataset.X_test)
            self.dataset.X_test = sequence.pad_sequences(sequences, maxlen=self.dataset.max_length,
                                                         padding=self.dataset.sequence_padding)

    def create_embedding_matrix(self):
        model = gensim.models.Word2Vec(self.dataset.dataset_df[self.dataset.data_column], vector_size=200, window=10,
                                       min_count=5, workers=10)
        model.train(self.dataset.dataset_df[self.dataset.data_column],
                    total_examples=len(self.dataset.dataset_df[self.dataset.data_column]), epochs=10)
        word_vectors = model.wv
        vocab_size = len(self.dataset.tokenizer.word_index) + 1

        embedding_matrix = np.zeros((vocab_size, 200))
        for word, index in self.dataset.tokenizer.word_index.items():
            if word_vectors.has_index_for(word):
                embedding_vector = word_vectors.get_vector(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        self.dataset.embedding = embedding_matrix

    def visualize(self):
        dataset_positive = self.dataset.dataset_df[self.dataset.dataset_df[self.dataset.label_column] == 1]
        dataset_negative = self.dataset.dataset_df[self.dataset.dataset_df[self.dataset.label_column] == 0]
        data = " ".join(tweet if not isinstance(tweet, list) else " ".join(tweet) for tweet in
                        self.dataset.dataset_df[self.dataset.data_column])
        data_positive = " ".join(tweet if not isinstance(tweet, list) else " ".join(tweet) for tweet in
                                 dataset_positive[self.dataset.data_column])
        data_negative = " ".join(tweet if not isinstance(tweet, list) else " ".join(tweet) for tweet in
                                 dataset_negative[self.dataset.data_column])

        wordcloud_fig, ax = plt.subplots(3, 1, figsize=(30, 30))

        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data)
        wordcloud_positive = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(
            data_positive)
        wordcloud_negative = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(
            data_negative)

        # Display the generated image:
        ax[0].imshow(wordcloud, interpolation='bilinear')
        ax[0].set_title('Data', fontsize=30)
        ax[0].axis('off')
        ax[1].imshow(wordcloud_positive, interpolation='bilinear')
        ax[1].set_title('Positive data', fontsize=30)
        ax[1].axis('off')
        ax[2].imshow(wordcloud_negative, interpolation='bilinear')
        ax[2].set_title('Negative data', fontsize=30)
        ax[2].axis('off')

        label_distribution_fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.countplot(self.dataset.dataset_df[self.dataset.label_column], ax=ax)

        self.dataset.save_visualization(wordcloud_fig, label_distribution_fig)
