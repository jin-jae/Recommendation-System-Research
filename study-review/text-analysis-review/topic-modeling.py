from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")

# 8 topics
cats = ["rec.motorcycles", "rec.sport.baseball", "comp.graphics",
        "comp.windows.x", "talk.politics.mideast",
        "soc.religion.christian", "sci.electronics", "sci.med"]

# extract only `cats` categories
news_df = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), categories=cats, random_state=0)

count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words="english", ngram_range=(1, 2))

feat_vect = count_vect.fit_transform(news_df.data)

lda = LatentDirichletAllocation(n_components=8, random_state=0)
lda.fit(feat_vect)

# print how much lda components have
# for each topic, how many word feature has been allocated to that topic
print(lda.components_.shape)


def display_topics(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print("topic #{0}".format(topic_index))

        # return array index of the biggest value
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes = topic_word_indexes[:no_top_words]

        # extract word feature for top_indexes and concat
        feature_concat = ' '.join([feature_names[i] for i in top_indexes])
        print(feature_concat)


# extract from get_feature_names()
feature_names = count_vect.get_feature_names()

# top 15
display_topics(lda, feature_names, 15)
