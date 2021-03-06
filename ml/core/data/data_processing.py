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

from ml.core.data.dataset import Dataset

le = LabelEncoder()
PUNCTUATION_LIST = list(string.punctuation)
stopword = nltk.corpus.stopwords.words('english')
tknzr = TweetTokenizer()
ps = PorterStemmer()
wn = WordNetLemmatizer()
tok = Tokenizer()


class DataProcessing:

    def __init__(self, dataset: Dataset, is_runtime: bool = False):
        self.__dataset: Dataset = dataset
        self.__is_runtime: bool = is_runtime
        self.PROCEDURES = {
            'drop_unused_columns': self.__drop_unused_columns,
            'encode_labels': self.encode_labels,
            'lowercase': self.__to_lowercase,
            'remove_urls': self.__remove_urls,
            'remove_urls_1': self.__remove_urls_1,
            'remove_placeholders': self.__remove_placeholders,
            'remove_placeholders_1': self.__remove_placeholders_1,
            'remove_frames': self.__remove_frames,
            'remove_html_references': self.__remove_html_references,
            'remove_non_letter_characters': self.__remove_non_letter_characters,
            'remove_mentions': self.__remove_mentions,
            'remove_mentions_1': self.__remove_mentions_1,
            'remove_digits': self.__remove_digits,
            'expand_contractions': self.__expand_contractions,
            'remove_punctuation': self.__remove_punctuation,
            'tokenize': self.__tokenize,
            'remove_stopwords': self.__remove_stopwords,
            'stem': self.__stem,
            'lemmatize': self.__lemmatize,
            'split': self.__split,
            'to_list': self.__to_list,
            'text_to_sequences': self.__text_to_sequences,
            'create_embedding_matrix': self.__create_embedding_matrix,
            'visualize': self.__visualize
        }

    def proceed(self):
        steps = self.__dataset.processing
        if self.__is_runtime:
            steps = self.__dataset.runtime_processing

        for procedure in steps:
            self.PROCEDURES.get(procedure)()

    def __drop_unused_columns(self):
        columns = deepcopy(self.__dataset.columns)
        columns.remove(self.__dataset.data_column)
        columns.remove(self.__dataset.label_column)
        self.__dataset.dataset_df = self.__dataset.dataset_df.drop(columns, axis=1)

    def encode_labels(self):
        self.__dataset.dataset_df[self.__dataset.label_column] = le.fit_transform(
            self.__dataset.dataset_df[self.__dataset.label_column])

    def __to_lowercase(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].str.lower()

    def __remove_urls(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'https?:\/\/\S+', self.__dataset.replace_character, text))
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", self.__dataset.replace_character, text))

    def __remove_urls_1(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'http?:\/\/\S+', self.__dataset.replace_character, text))
        self.__remove_urls()

    def __remove_placeholders(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'{link}', self.__dataset.replace_character, text))
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r"\[video\]", self.__dataset.replace_character, text))

    def __remove_placeholders_1(self):
        self.__remove_placeholders()
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'<\w+>', self.__dataset.replace_character, text))

    def __remove_frames(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'"\d+x\d+ custom picture frame / poster frame (\d+|\d+\.\d+) "',
                                self.__dataset.replace_character, text))
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'frame \( \d+ this frame is', self.__dataset.replace_character, text))

    def __remove_html_references(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'&[a-z]+;', self.__dataset.replace_character, text))

    def __remove_non_letter_characters(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", self.__dataset.replace_character, text))

    def __remove_mentions(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'@mention', self.__dataset.replace_character, text))

    def __remove_mentions_1(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'@\w+', self.__dataset.replace_character, text))

    def __remove_digits(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: re.sub(r'\d+', self.__dataset.replace_character, text))

    def __expand_contractions(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: contractions.fix(text))

    def __remove_punctuation(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: "".join(
                [char if char not in PUNCTUATION_LIST else self.__dataset.replace_character for char in text]))

    def __tokenize(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            tknzr.tokenize)

    def __remove_stopwords(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: [word for word in text if word not in stopword])

    def __stem(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: [ps.stem(word) for word in text])

    def __lemmatize(self):
        self.__dataset.dataset_df[self.__dataset.data_column] = self.__dataset.dataset_df[
            self.__dataset.data_column].apply(
            lambda text: [wn.lemmatize(word) for word in text])

    def __split(self):
        self.__dataset.X_train, self.__dataset.X_test, self.__dataset.Y_train, self.__dataset.Y_test = \
            train_test_split(self.__dataset.dataset_df[self.__dataset.data_column],
                             self.__dataset.dataset_df[self.__dataset.label_column],
                             test_size=self.__dataset.test_ratio)

    def __to_list(self):
        self.__dataset.X_train = self.__dataset.dataset_df[self.__dataset.data_column]

    def __text_to_sequences(self):
        self.__dataset.tokenizer = Tokenizer()
        self.__dataset.tokenizer.fit_on_texts(self.__dataset.X_train)

        sequences = self.__dataset.tokenizer.texts_to_sequences(self.__dataset.X_train)
        self.__dataset.X_train = sequence.pad_sequences(sequences, maxlen=self.__dataset.max_length,
                                                        padding=self.__dataset.sequence_padding)

        if self.__dataset.X_test is not None:
            sequences = self.__dataset.tokenizer.texts_to_sequences(self.__dataset.X_test)
            self.__dataset.X_test = sequence.pad_sequences(sequences, maxlen=self.__dataset.max_length,
                                                           padding=self.__dataset.sequence_padding)

    def __create_embedding_matrix(self):
        model = gensim.models.Word2Vec(self.__dataset.dataset_df[self.__dataset.data_column], vector_size=200,
                                       window=10,
                                       min_count=5, workers=10)
        model.train(self.__dataset.dataset_df[self.__dataset.data_column],
                    total_examples=len(self.__dataset.dataset_df[self.__dataset.data_column]), epochs=10)
        word_vectors = model.wv
        vocab_size = len(self.__dataset.tokenizer.word_index) + 1

        embedding_matrix = np.zeros((vocab_size, 200))
        for word, index in self.__dataset.tokenizer.word_index.items():
            if word_vectors.has_index_for(word):
                embedding_vector = word_vectors.get_vector(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        self.__dataset.embedding = embedding_matrix

    def __visualize(self):
        dataset_positive = self.__dataset.dataset_df[self.__dataset.dataset_df[self.__dataset.label_column] == 1]
        dataset_negative = self.__dataset.dataset_df[self.__dataset.dataset_df[self.__dataset.label_column] == 0]
        data = " ".join(tweet if not isinstance(tweet, list) else " ".join(tweet) for tweet in
                        self.__dataset.dataset_df[self.__dataset.data_column])
        data_positive = " ".join(tweet if not isinstance(tweet, list) else " ".join(tweet) for tweet in
                                 dataset_positive[self.__dataset.data_column])
        data_negative = " ".join(tweet if not isinstance(tweet, list) else " ".join(tweet) for tweet in
                                 dataset_negative[self.__dataset.data_column])

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
        sns.countplot(self.__dataset.dataset_df[self.__dataset.label_column], ax=ax)

        self.__dataset.save_visualization(wordcloud_fig, label_distribution_fig)
