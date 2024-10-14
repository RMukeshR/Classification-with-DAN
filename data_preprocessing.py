## Read data

df_pos = open("Train.pos","r", encoding= "latin-1").read()
df_neg = open("Train.neg","r", encoding= "latin-1").read()
df_test = open("TestData","r", encoding= "latin-1").read()

df_pos_list = [i for i in df_pos.split("\n") if len(i) >= 2]
df_neg_list = [i for i in df_neg.split("\n") if len(i) >= 2]
df_test_list = [i for i in df_test.split("\n") if len(i) >= 2]


# import gensim.downloader as api
from gensim.models import KeyedVectors


# Load and save pre-trained Word2Vec model
# word2vec_model = api.load("word2vec-google-news-300")
# word2vec_model.save("word2vec_model.bin")


# Load and save the pre-trained GloVe model
# glove_model = api.load("glove-wiki-gigaword-300")
# glove_model.save("glove_model.bin")


# # Load and save the pre-trained FastText model
# fasttext_model = api.load("fasttext-wiki-news-subwords-300")
# fasttext_model.save("fasttext_model.bin")

# Load the saved Word2Vec model from file
loaded_woord2vec_model = KeyedVectors.load("word2vec_model.bin")
loaded_glove_model = KeyedVectors.load("glove_model.bin")
loaded_fasttext_model = KeyedVectors.load("fasttext_model.bin")


# Find similar words to "apple" using the loaded model
similar_words_word2vec = loaded_woord2vec_model.most_similar("apple", topn=5)
similar_words_glove = loaded_glove_model.most_similar("apple", topn=5)
similar_words_fasttext = loaded_fasttext_model.most_similar("apple", topn=5)


# print("Words similar to 'apple':")
for word, similarity in similar_words_word2vec:
    print(f"{word}: {similarity:.4f}")

for word, similarity in similar_words_glove:
    print(f"{word}: {similarity:.4f}")

for word, similarity in similar_words_fasttext:
    print(f"{word}: {similarity:.4f}")