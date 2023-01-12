import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import *
import string
import sys


def contains_nums(text):
    if any(char.isdigit() for char in text):
        return True
    else:
        return False


def preprocess(file):
    df1 = pd.read_csv(file)

    df1 = df1.drop_duplicates()

    length_of_df = df1.shape[0] - 1
    count = 0
    all_comments = []
    all_sent = df1['Sentiment']

    sw = list(stopwords.words())
    stemmer = PorterStemmer()

    for rev in df1['Review']:
        rev = rev.lower()
        rev_tokens = wordpunct_tokenize(rev)
        rev_clean = [word for word in rev_tokens if word not in sw and not contains_nums(word) and
                     word not in string.punctuation]
        stemmed = [stemmer.stem(word) for word in rev_clean]
        rev_combined = " ".join(stemmed)
        sys.stdout.write('\r')
        sys.stdout.write(f'Processing {file}... {round(count / length_of_df * 100, 0)}%')
        sys.stdout.flush()
        count += 1
        all_comments.append(rev_combined)
    sys.stdout.write('\r')
    sys.stdout.write(f'Processing of {file} complete')

    d = {'Sentiment': all_sent, 'Reviews': all_comments}

    return pd.DataFrame(d)


if __name__ == "__main__":
    df = preprocess('car-reviews.csv')
