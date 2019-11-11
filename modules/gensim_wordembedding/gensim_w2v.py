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

class gensim_mode:
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

    def train_csv(self):
        pass

    def train_sent_list(self,sentences):
        w2v_model = Word2Vec(min_count=100,
                             window=5,
                             size=100,
                             workers=4)

        w2v_model.build_vocab(sentences, progress_per=10000)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count,
                        epochs=30, report_delay=1)
        w2v_model.init_sims(replace=True)

        # print(w2v_model.wv.most_similar(positive=['economy']))
        # print(w2v_model.wv.similarity('video', 'gaming'))
        model_path = "/media/ekbana/ekbana500/NLP TEAM/mygithub/word-embeddings/models"
        pickle.dump(w2v_model, open(model_path, 'wb'))

        self.tsne_plot(w2v_model)
        return w2v_model

    def train_text(self):
        sentences = word2vec.Text8Corpus(
            '/media/ekbana/ekbana500/NLP TEAM/mygithub/word-embeddings/data/nepali_w2v.txt')
        model = word2vec.Word2Vec(sentences, size=200, window=3, min_count=5)
        model.save(
            '/media/ekbana/ekbana500/NLP TEAM/mygithub/word-embeddings/models/nep_word2Vec_small.w2v')
        return model

    def load_googlebin(self):
        gensim_model = "/media/ekbana/ekbana500/Pema/datasets/GoogleNews-vectors-negative300.bin.gz"
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(
            gensim_model, binary=True, limit=100000)
        gensim_vector = gensim_model['dog']
        print(gensim_vector.shape)
        print(gensim_vector)

    def load_savedModel(self,filename):
        model = gensim.models.KeyedVectors.load(filename)
        result = model.most_similar(positive=['कम्पनी', 'स्कूल'],
                                    negative=['कम्पनी'], topn=1)
        print(result)


    def load_pickle(self,model_path):
        model = pickle.load(open(model_path, 'rb'))
        return model

    def test(self):
        df = pd.read_csv('/media/ekbana/ekbana500/NLP TEAM/mygithub/word-embeddings/data/bbc-text.csv')
        df_clean = pd.DataFrame(df.text.apply(lambda x: self.clean_text(x)))
        nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

        df_clean["text_lemmatize"] =  df_clean.apply(lambda x: self.lemmatizer(x['text'],nlp), axis=1)
        df_clean['text_lemmatize_clean'] = df_clean['text_lemmatize'].str.replace('-PRON-', '')

        sentences = [row.split() for row in df_clean['text_lemmatize_clean']]
        w2v_model = self.train_sent_list(sentences)
        print(w2v_model.wv.most_similar(positive=['economy']))
        print(w2v_model.wv.similarity('video', 'gaming'))



c = gensim_mode()
# c.test()
# c.train_text()
c.load_savedModel('/media/ekbana/ekbana500/NLP TEAM/mygithub/word-embeddings/models/nep_word2Vec_small.w2v')





