import nltk
from nltk.corpus import brown
import random
from nltk.corpus import stopwords

def create_news_or_not():
    news_or_not =[]
    for category in brown.categories():
        for fileid in brown.fileids(category):
            if category == 'news':
                news_or_not.append((brown.words(fileid), category))
            else:
                news_or_not.append((brown.words(fileid), 'non-news'))
    return news_or_not

news_or_not = create_news_or_not()
random.shuffle(news_or_not)
feature_words = nltk.FreqDist(w.lower() for w in brown.words() if w not in stopwords.words('english') and w.isalnum())
word_features = list(feature_words)[:2000]


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


featuresets = [(document_features(d), c) for (d, c) in news_or_not]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
most_informative_words = [w[9:-1] for (w,b) in classifier.most_informative_features(300)]
print(nltk.classify.accuracy(classifier, test_set))

random.shuffle(news_or_not)

def document_features_most_informative(document):
    document_words = set(document)
    features = {}
    for word in most_informative_words:
        features['contains({})'.format(word)] = (word in document_words)
    return features
featuresets = [(document_features_most_informative(d), c) for (d,c) in news_or_not]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(30))

