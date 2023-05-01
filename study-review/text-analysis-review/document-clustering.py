import glob
import os
import warnings

import pandas as pd

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 700)

path = r"/Users/jinjae/Code/Study/Python-Machine-Learning/OpinosisDataset1.0/topics"
# all .data files to a list
all_files = glob.glob(os.path.join(path, "*.data"))
filename_list = []
opinion_text = []

# load with DataFrame and convert to string, merge to opinion_text list
for file_ in all_files:
    # to DataFrame
    df = pd.read_table(file_, index_col=None, header=0, encoding="latin1")
    print("df.to_string here:", df.to_string())

    filename_ = file_.split('/')[-1]
    filename = filename_.split('.')[0]

    filename_list.append(filename)
    opinion_text.append(df.to_string())

document_df = pd.DataFrame({"filename": filename_list, "opinion_text": opinion_text})

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()


# 입력으로 들어온 token단어들에 대해서 lemmatization 어근 변환.
def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]


# TfidfVectorizer 객체 생성 시 tokenizer인자로 해당 함수를 설정하여 lemmatization 적용
# 입력으로 문장을 받아서 stop words 제거-> 소문자 변환 -> 단어 토큰화 -> lemmatization 어근 변환.
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english",
                             ngram_range=(1, 2), min_df=0.05, max_df=0.85)

feature_vect = tfidf_vect.fit_transform(document_df["opinion_text"])

from sklearn.cluster import KMeans

km_cluster = KMeans(n_clusters=5, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_

document_df["cluster_label"] = cluster_label

print(document_df[document_df["cluster_label"] == 0].sort_values(by="filename"))
