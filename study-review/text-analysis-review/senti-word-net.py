import nltk
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

nltk.download("all")


# PennTreebank Tag base
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB


def swn_polarity(text):
    sentiment = 0.0
    tokens_count = 0

    lemmatizer = WordNetLemmatizer()
    raw_sentences = sent_tokenize(text)
    # for each sentences, generate SentiSynset -> add all
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            # lemmatize
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            # make Synset object based on word and part of speech
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            # calculate with positive: +, negative: -
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += (swn_synset.pos_score() - swn_synset.neg_score())
            tokens_count += 1

    if not tokens_count:
        return 0

    if sentiment >= 0 :
        return 1
    return 0


review_df = pd.read_csv("../../dataset/etc/labeledTrainData.tsv", header=0, sep="\t", quoting=3)

review_df["preds"] = review_df["review"].apply(lambda x: swn_polarity(x))
y_target = review_df["sentiment"].values
preds = review_df["preds"].values

print(confusion_matrix(y_target, preds))
print("accuracy:", np.round(accuracy_score(y_target, preds), 4))
print("precision:", np.round(precision_score(y_target, preds), 4))
print("recall:", np.round(recall_score(y_target, preds), 4))
