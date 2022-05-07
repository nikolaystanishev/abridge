import csv

import pandas as pd

# neg = './acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/train_neg_full.txt'
# pos = './acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/train_pos_full.txt'
# output = './acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/full.tsv'
neg = './acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/train_neg.txt'
pos = './acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/twitter-datasets/train_pos.txt'
output = './acfe063c-c5b9-4aa6-b5fb-12c88bae9627_epfml-text/partial.tsv'

neg_dataset_df = pd.read_csv(neg, names=['data'], sep='\t', encoding="latin-1", quoting=csv.QUOTE_NONE)
pos_dataset_df = pd.read_csv(pos, names=['data'], sep='\t', encoding="latin-1", quoting=csv.QUOTE_NONE)

neg_dataset_df.info()
pos_dataset_df.info()

neg_dataset_df['label'] = 0
pos_dataset_df['label'] = 1

full_dataset_df = pd.concat([neg_dataset_df, pos_dataset_df])
full_dataset_df = full_dataset_df.drop_duplicates(subset=['data'], keep='last')

full_dataset_df.to_csv(output, sep='\t', index=False)
