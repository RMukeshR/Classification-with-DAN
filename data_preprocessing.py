import gensim.downloader as api

# Load and save pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")
word2vec_model.save("word2vec_model.bin")