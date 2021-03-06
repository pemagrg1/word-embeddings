import gensim
from gensim.models import Word2Vec,word2vec
import pickle
import spacy
import re, string
import pandas as pd
from collections import defaultdict
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from matplotlib import pyplot as plt

STOPWORDS = set(stopwords.words('english'))

class gensim_model:
    def __init__(self):
        pass

    def tsne_plot(self,model):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca',
                          n_iter=2500,
                          random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(18, 18))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()

    def clean_text(self,text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        if len(text) > 2:
            return ' '.join(
                word for word in text.split() if word not in STOPWORDS)

    def lemmatizer(self,text,nlp):
        sent = []
        doc = nlp(text)
        for word in doc:
            sent.append(word.lemma_)
        return " ".join(sent)

    def create_embedding(self,sentences):
        w2v_model = Word2Vec(min_count=100,
                             window=5,
                             size=100,
                             workers=4)

        w2v_model.build_vocab(sentences, progress_per=10000)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count,
                        epochs=30, report_delay=1)
        w2v_model.init_sims(replace=True)

        model_path = "word-embeddings/models/en.model"
        pickle.dump(w2v_model, open(model_path, 'wb'))
        self.tsne_plot(w2v_model)
        return w2v_model


    def gensim_embedding(self,type,data_or_path):
        """
        :param type:
            text8 : if you have a text file with full text, you can pass type as "text" and the text file path.
            google bin: you can download the google bin file and pass type as google_bin and the path.
            sentence_list: you can train your own list of sentences.
            pickle: if saved pickle file is there, you can pass type as pickle and the path.
            csv : in the csv, "text" column is taken for word embedding.
        :param data:
            text8 : 'data/en_w2v.txt'
            google bin: "GoogleNews-vectors-negative300.bin.gz"
            sentence_list: list of sentences
            pickle: the saved pickle file after training.
            csv: 'data/bbc-text.csv'
        :return: model
        """
        model = None

        if type == "text":
            sentences = word2vec.Text8Corpus(data_or_path)
            model = self.create_embedding(sentences)

        elif type == "sentence_list":
            sentences = data_or_path
            model = self.create_embedding(sentences)

        elif type == "google_bin":
            gensim_model = data_or_path
            model = gensim.models.KeyedVectors.load_word2vec_format(
                gensim_model, binary=True, limit=100000)

        elif type == "pickle":
            model = gensim.models.KeyedVectors.load(data_or_path)

        elif type == "csv":
            df = pd.read_csv(data_or_path)
            df_clean = pd.DataFrame(
                df.text.apply(lambda x: self.clean_text(x)))
            nlp = spacy.load('en', disable=['ner',
                                            'parser'])  # disabling Named Entity Recognition for speed
            df_clean["text_lemmatize"] = df_clean.apply(
                lambda x: self.lemmatizer(x['text'], nlp), axis=1)
            df_clean['text_lemmatize_clean'] = df_clean[
                'text_lemmatize'].str.replace('-PRON-', '')
            sentences = [row.split() for row in
                         df_clean['text_lemmatize_clean']]
            model = self.create_embedding(sentences)

        print(model.wv.most_similar(positive=['economy']))
        print(model.wv.similarity('video', 'gaming'))
        return model


# TEST
#
# c = gensim_model()
# c.gensim_embedding(type="csv",data_or_path="word-embeddings/data/bbc-text.csv")
# c.gensim_embedding(type="pickle",data_or_path="word-embeddings/models/en.model")
