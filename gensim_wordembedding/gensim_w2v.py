import gensim
gensim_model = "/media/ekbana/ekbana500/Pema/datasets/GoogleNews-vectors-negative300.bin.gz"
gensim_model = gensim.models.KeyedVectors.load_word2vec_format(gensim_model, binary=True, limit=100000)
gensim_vector = gensim_model['dog']
print(gensim_vector.shape)
print (gensim_vector)
# ##for 2D
# test = gensim_vector.reshape(1, -1)
# print (test)