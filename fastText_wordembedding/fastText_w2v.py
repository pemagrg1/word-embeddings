import fasttext
import numpy as np

"""
    download the pretrained model from: https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md
"""

model_fasttext = fasttext.load_model("wiki.en/wiki.en.bin")
n_features = model_fasttext.get_dimension()
window_length = 200

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    words = text.split()
    window = words[-window_length:]

    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = model_fasttext.get_word_vector(word).astype('float32')

    return x

print (text_to_vector("queen"))
print ()
print (model_fasttext["queen"])