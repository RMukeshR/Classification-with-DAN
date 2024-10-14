## Read data

df_pos = open("Train.pos","r", encoding= "latin-1").read()
df_neg = open("Train.neg","r", encoding= "latin-1").read()
df_test = open("TestData","r", encoding= "latin-1").read()

df_pos_list = [i for i in df_pos.split("\n") if len(i) >= 2]
df_neg_list = [i for i in df_neg.split("\n") if len(i) >= 2]
df_test_list = [i for i in df_test.split("\n") if len(i) >= 2]


import gensim.downloader as api
from gensim.models import KeyedVectors


# Load and save pre-trained Word2Vec model
# word2vec_model = api.load("word2vec-google-news-300")
# word2vec_model.save("word2vec_model.bin")


# Load the saved Word2Vec model from file
loaded_model = KeyedVectors.load("word2vec_model.bin")

# Find similar words to "apple" using the loaded model
similar_words = loaded_model.most_similar("apple", topn=5)

print("Words similar to 'apple':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")