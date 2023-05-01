from nltk.stem import WordNetLemmatizer
import nltk
import string

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()


# 입력으로 들어온 token단어들에 대해서 lemmatization 어근 변환.
def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]


# TfidfVectorizer 객체 생성 시 tokenizer인자로 해당 함수를 설정하여 lemmatization 적용
# 입력으로 문장을 받아서 stop words 제거-> 소문자 변환 -> 단어 토큰화 -> lemmatization 어근 변환.
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


import pandas as pd
import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings

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

    filename_ = file_.split('/')[-1]
    filename = filename_.split('.')[0]

    filename_list.append(filename)
    opinion_text.append(df.to_string())

document_df = pd.DataFrame({"filename": filename_list, "opinion_text": opinion_text})

tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english",
                             ngram_range=(1, 2), min_df=0.05, max_df=0.85)

feature_vect = tfidf_vect.fit_transform(document_df["opinion_text"])

km_cluster = KMeans(n_clusters=3, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_
document_df["cluster_label"] = cluster_label

from sklearn.metrics.pairwise import cosine_similarity

hotel_indexes = document_df[document_df["cluster_label"] == 2].index
print("Dataframe Index:", hotel_indexes)

comparison_docname = document_df.iloc[hotel_indexes[0]]["filename"]
print("###", comparison_docname, "and other document similarities")

similarity_pair = cosine_similarity(feature_vect[hotel_indexes[0]], feature_vect[hotel_indexes])
print(similarity_pair)

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sorted_index = similarity_pair.argsort()[:, ::-1]
sorted_index = sorted_index[:, 1:]

# resort
hotel_sorted_indexes = hotel_indexes[sorted_index.reshape(-1)]

hotel_1_sim_value = np.sort(similarity_pair.reshape(-1))[::-1]
hotel_1_sim_value = hotel_1_sim_value[1:]

# visualize
hotel_1_sim_df = pd.DataFrame()
hotel_1_sim_df['filename'] = document_df.iloc[hotel_sorted_indexes]['filename']
hotel_1_sim_df['similarity'] = hotel_1_sim_value

print("most similarity filename and similarity:\n", hotel_1_sim_df.iloc[0, :])
sns.barplot(x='similarity', y='filename', data=hotel_1_sim_df)
plt.title(comparison_docname)


## 중요!