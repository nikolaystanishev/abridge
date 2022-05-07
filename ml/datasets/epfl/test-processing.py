# Relevant imports
import re

import gensim
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import asarray
from numpy import zeros
from sklearn.model_selection import train_test_split

SEED = 15432


def replace_exclamations(x):
    """ Replaces multiple exclamation marks by the word exclamationMark """
    x = re.sub('(\! )+(?=(\!))', '', x)
    x = re.sub(r"(\!)+", ' exclamationMark ', x)
    return x


def replace_points(x):
    """ Replaces multiple points by the word multiplePoints """
    x = re.sub('(\. )+(?=(\.))', '', x)
    x = re.sub(r"(\.)+", ' multistop ', x)
    return x


def replace_questions(x):
    """ Replaces multiple question marks by the word questionMark """
    x = re.sub('(\? )+(?=(\?))', '', x)
    x = re.sub(r"(\?)+", ' questionMark ', x)
    return x


def tokenization(text):
    text = re.split('\W+', text)
    return text


def lemmatizer(l, text):
    text = [l.lemmatize(word) for word in text]
    return text


def join_tokens(tokens):
    text = ' '.join(tokens)
    return text


def translate_emojis(x):
    """ Replace emojis into meaningful words """

    x = re.sub(' [:,=,8,;]( )*([\',\"])*( )*(-)*( )*[\),\],},D,>,3,d] ', ' happy ', x)  #:) :D :} :] :> :') :'D :-D :-)
    x = re.sub(' [\(,\[,{,<]( )*([\',\"])*( )*[:,=,8,;] ', ' happy ', x)  # inverted happy
    x = re.sub(' [X,x]( )*D ', ' funny ', x)  # XD xD
    x = re.sub(' ^( )*[.,~]( )*^ ', ' happy ', x)  # ^.^  ^ ~ ^
    x = re.sub(' ^( )*(_)+( )*^ ', ' happy ', x)  # ^__^

    x = re.sub(' [:,=,8,;]( )*([\',\"])*( )*(-)*( )*[\(,\[,{,<] ', ' sad ', x)  #:( :{ :[ :< :'(
    x = re.sub(' [\),\],},D,>,d]( )*(-)*( )*([\',\"])*( )*[:,=,8,;] ', ' sad ', x)  # inverted sad
    x = re.sub(' >( )*.( )*< ', ' sad ', x)  # >.<
    x = re.sub(' <( )*[ \/,\\ ]( )*3 ', ' sad ', x)  # </3

    x = re.sub(' [:,=,8]( )*(-)*( )*p ', ' silly ', x)  #:p :-p
    x = re.sub(' q( )*(-)*( )*[:,=,8] ', ' silly ', x)  # silly inverted

    x = re.sub(' [:,=,8]( )*$ ', ' confused ', x)  # =$ 8$
    x = re.sub(' [:,=,8]( )*@ ', ' mad ', x)  #:@
    x = re.sub(' [:,=,8]( )*(-)*( )*[\/,\\,|] ', ' confused ', x)  #:/ :\
    x = re.sub(' [\/,\\,|]( )*(-)*( )*[:,=,8] ', ' confused ', x)  # confused inverted

    x = re.sub(' [:,=,8,;]( )*(-)*( )*[o,0] ', ' surprised ', x)  #:o :-O

    x = re.sub(' [x,X]+ ', ' kiss ', x)  # xXxX
    x = re.sub(' ([x,X][o,O]){2,} ', ' kiss ', x)  # xoxo
    x = re.sub(' [:,=,8,;]( )*\* ', ' kiss ', x)  #:* =*
    x = re.sub(' <( )*3 ', ' love ', x)  # <3

    x = re.sub('#', ' hashtag ', x)  # hashtag
    x = re.sub('&', ' and ', x)  # &
    x = re.sub(' \(( )*y( )*\) ', ' yes ', x)  # (y)
    x = re.sub(' w( )*/ ', ' without ', x)  # w/

    x = re.sub(' ([h,j][a,e,i,o]){2,} ', ' haha ', x)  # hahah
    x = re.sub(' (a*ha+h[ha]*|h*ah+a[ah]*|o?l+o+l+[ol]*) ', ' haha ', x)  # hhaha,aahha,lool
    x = re.sub(' (i*hi+h[hi]*|h*ih+i[ih]*|h*oh+o[oh]*|h*eh+e[eh]*) ', ' haha ', x)  # hihi, hoho, hehe
    return x


def split_negation(text):
    negations_dict = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                      "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                      "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                      "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                      "mustn't": "must not", "ain't": "is not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dict.keys()) + r')\b')
    text = neg_pattern.sub(lambda x: negations_dict[x.group()], text)
    return text


def replace_contractions(text):
    contractions_dict = {"i'm": "i am", "wanna": "want to", "whi": "why", "gonna": "going to",
                         "wa": "was", "nite": "night", "there's": "there is", "that's": "that is",
                         "ladi": "lady", "fav": "favorite", "becaus": "because", "i\'ts": "it is",
                         "dammit": "damn it", "coz": "because", "ya": "you", "dunno": "do not know",
                         "donno": "do not know", "donnow": "do not know", "gimme": "give me"}
    contraction_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    text = contraction_pattern.sub(lambda x: contractions_dict[x.group()], text)

    contraction_patterns = [(r'ew(\w+)', 'disgusting'), (r'argh(\w+)', 'argh'), (r'fack(\w+)', 'fuck'),
                            (r'sigh(\w+)', 'sigh'), (r'fuck(\w+)', 'fuck'), (r'omg(\w+)', 'omg'),
                            (r'oh my god(\w+)', 'omg'), (r'(\w+)n\'', '\g<1>ng'), (r'(\w+)n \'', '\g<1>ng'),
                            (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'),
                            (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'),
                            ('y+a+y+', 'yay'), ('y+[e,a]+s+', 'yes'), ('n+o+', 'no'), ('a+h+', 'ah'),
                            ('m+u+a+h+', 'kiss'),
                            (' y+u+p+ ', ' yes '), (' y+e+p+ ', ' yes '), (' idk ', ' i do not know '),
                            (' ima ', ' i am going to '),
                            (' nd ', ' and '), (' dem ', ' them '), (' n+a+h+ ', ' no '), (' n+a+ ', ' no '),
                            (' w+o+w+', 'wow '),
                            (' w+o+a+ ', ' wow '), (' w+o+ ', ' wow '), (' a+w+ ', ' cute '), (' lmao ', ' haha '),
                            (' gad ', ' god ')]
    patterns = [(re.compile(regex_exp, re.IGNORECASE), replacement)
                for (regex_exp, replacement) in contraction_patterns]
    for (pattern, replacement) in patterns:
        (text, _) = re.subn(pattern, replacement, text)
    return text


def tweet_cleaner(tweet):
    # Add trailing and leading whitespaces for the sake of preprocessing
    tweet = ' ' + tweet + ' '
    # translate emojis
    tweet = translate_emojis(tweet)
    # lowercase tweet
    tweet = tweet.lower()
    # seperate negation words
    tweet = split_negation(tweet)
    # seperate punctuation from words
    tweet = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", tweet)
    # remove the observed pattern of numbers seen above
    tweet = re.sub(r'\\ [0-9]+ ', '', tweet)
    # replace ?,!,. by words
    tweet = replace_exclamations(tweet)
    tweet = replace_questions(tweet)
    tweet = replace_points(tweet)

    # Now since we translated punctuation and emojis and negative words we can remove the rest of the 'unwanted' chars
    # remove unwanted punctuation
    tweet = re.sub("[^a-zA-Z]", " ", tweet)

    # remove trailing and leading whitespace
    tweet = tweet.strip()
    # remove multiple consecutive whitespaces
    tweet = re.sub(' +', ' ', tweet)

    # Lemmatization
    l = WordNetLemmatizer()
    tweet = tokenization(tweet)
    tweet = join_tokens(lemmatizer(l, tweet))
    return tweet


def train_test_cleaner():
    """Clean train set and test set and return cleaned dataframes"""
    # Read positive tweets train file
    with open('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/train_pos_full.txt', "r",
              encoding="utf8") as file:
        train_pos = file.read().split('\n')
    train_pos = pd.DataFrame({'tweet': train_pos})[:len(train_pos) - 1]
    # Read negative tweets train file
    with open('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/train_neg_full.txt', "r",
              encoding="utf8") as file:
        train_neg = file.read().split('\n')
    train_neg = pd.DataFrame({'tweet': train_neg})[:len(train_neg) - 1]
    # Read test tweets file
    with open('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/test_data.txt', "r",
              encoding="utf8") as file:
        df_unknown = file.read().split('\n')
    df_unknown = pd.DataFrame({'tweet': df_unknown})[:len(df_unknown) - 1]
    df_unknown.index += 1
    df_unknown['tweet'] = df_unknown['tweet'].apply(lambda x: str(x).split(',', maxsplit=1)[1])

    # Drop duplicates
    train_neg.drop_duplicates(inplace=True)
    train_pos.drop_duplicates(inplace=True)

    # Add labels
    train_pos['label'] = 1
    train_neg['label'] = 0
    train_set = train_pos.append(train_neg)

    # Apply tweet cleaner to train set and test
    train_set['tweet'] = train_set['tweet'].apply(tweet_cleaner)
    df_unknown['tweet'] = df_unknown['tweet'].apply(tweet_cleaner)

    return train_set, df_unknown


def tok_and_pad_for_nn(X_train, X_test, maxlen, num_words):
    """
    Tokenizes and pads to maxlen each tweet for each given dataframe
    so that it can directly be used in a neural network
    INPUT:
        X_train: Panda Series       - tweets
        X_test: Panda Series        - tweets
        to_predict: Panda Series    - tweets we have to label to submit to aicrowd
        maxlen: int                 - maximal length of word we take per tweet
        num_words: int              - the maximum number of words to keep, based on word frequency

    OUTPUT:
        X_train: ndarray
        X_test: ndarray
        to_predict: ndarray
        vocab_size: int
    """
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)  # convert each word to a integer based on the tokenizer
    X_test = tokenizer.texts_to_sequences(X_test)
    # to_predict = tokenizer.texts_to_sequences(to_predict)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)  # makes sure all tweets have 100 words (padding)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    # to_predict = pad_sequences(to_predict, padding='post', maxlen=maxlen)

    return X_train, X_test, vocab_size, tokenizer


def create_and_save_embedding_vect(df):
    """
    Creates and train a Word2Vec model from our corpus of words
    """

    # df = pd.read_csv("./data/tweets_full.csv")
    # df = df['tweet']

    # tokenize the tweets to get a list of the list of the words of each tweet
    l_tweets = list(df['tweet'].apply(lambda x: word_tokenize(x)))

    # size = vector dimension for each word
    # window = size of window neighborhood
    # min_count = need the word to appear at least that number to be taken in the model
    # workers = number of cores used to run the model

    model = gensim.models.Word2Vec(l_tweets, vector_size=200, window=10, min_count=5, workers=10)
    model.train(l_tweets, total_examples=len(l_tweets), epochs=10)
    word_vectors = model.wv  # get only the word embedding vectors from the model
    word_vectors.save_word2vec_format('embedding_vec.txt', binary=False)  # export the word embeddings vectors


def create_embedding_matrix(vocab_size, tokenizer):
    embeddings_dictionary = dict()
    glove_file = open('embedding_vec.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dim = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dim
    glove_file.close()

    embedding_matrix = zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


MAXLEN_140 = 140
MAXLEN_40 = 40
NUM_WORD = 120000

# load and clean the dataset
train_set, unknown = train_test_cleaner()
# Shuffle the order of the entries of the train set (to mix the labels)
train_set.sample(frac=1, random_state=1).reset_index(drop=True)
# create the word2vec model
create_and_save_embedding_vect(train_set)

# split training set
X_train, X_test, y_train, y_test = train_test_split(train_set['tweet'], train_set['label'], test_size=0.05,
                                                    random_state=42)

# prepare input for neural nets
X_train_140, X_test_140, vocab_size, tokenizer_140 = tok_and_pad_for_nn(X_train, X_test, MAXLEN_140, NUM_WORD)
X_train_40, X_test_40, vocab_size, tokenizer_40 = tok_and_pad_for_nn(X_train, X_test, MAXLEN_40, NUM_WORD)

# create embedding matrix from our word2vec model
embedding_matrix_140 = create_embedding_matrix(vocab_size, tokenizer_140)
embedding_matrix_40 = create_embedding_matrix(vocab_size, tokenizer_40)

np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-40/X_train.npy', X_train_40)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-40/Y_train.npy', y_train)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-40/X_test.npy', X_test_40)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-40/Y_test.npy', y_test)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-40/embedding.npy', embedding_matrix_40)

np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-140/X_train.npy', X_train_140)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-140/Y_train.npy', y_train)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-140/X_test.npy', X_test_140)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-140/Y_test.npy', y_test)
np.save('./acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/processed/test-140/embedding.npy', embedding_matrix_140)
