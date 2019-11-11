import fasttext
bin_path = "word-embeddings/bin_files/"
def train_Skipgram(text_file):
    # Skipgram model :
    model = fasttext.train_unsupervised(text_file, model='skipgram')

    print(model.words)   # list of words in dictionary
    print(model['king']) # get the vector of the word 'king'
    model.save_model(bin_path+text_file.split("/")[-1].split(".")[0]+"SkipgramfastText.bin")


def train_cbow(text_file):
    model = fasttext.train_unsupervised(text_file, model='cbow')

    print(model.words)   # list of words in dictionary
    print(model['king']) # get the vector of the word 'king'
    model.save_model(bin_path+text_file.split("/")[-1].split(".")[0]+"cbowfastText.bin")


def test(model_path):
    model = fasttext.load_model(model_path)
    print(model['king'])


train_Skipgram('word-embeddings/data/test_en_txt.txt')
test(bin_path+'test_en_txtSkipgramfastText.bin')
