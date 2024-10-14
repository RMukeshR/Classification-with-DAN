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
# loaded_woord2vec_model = KeyedVectors.load("word2vec_model.bin")
# loaded_glove_model = KeyedVectors.load("glove_model.bin")
# loaded_fasttext_model = KeyedVectors.load("fasttext_model.bin")


# # Find similar words to "apple" using the loaded model
# similar_words_word2vec = loaded_woord2vec_model.most_similar("apple", topn=5)
# similar_words_glove = loaded_glove_model.most_similar("apple", topn=5)
# similar_words_fasttext = loaded_fasttext_model.most_similar("apple", topn=5)


# print("Words similar to 'apple':")
# for word, similarity in similar_words_word2vec:
#     print(f"{word}: {similarity:.4f}")

# for word, similarity in similar_words_glove:
#     print(f"{word}: {similarity:.4f}")

# for word, similarity in similar_words_fasttext:
#     print(f"{word}: {similarity:.4f}")


## Data pre-processing

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a function to expand common English contractions
contractions = {
    "it’s": "it is", "it's": "it is", "don't": "do not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "we're": "we are", "they're": "they are", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mustn't": "must not"
}

def expand_contractions(text):
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()

    # 2. Expand contractions
    text = expand_contractions(text)

    # 3. Removing URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Removing Punctuation and Special Characters (but keeping spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # 5. Tokenization using Python's split instead of nltk word_tokenize
    tokens = text.split()

    # 6. Remove non-alphabetic tokens (e.g., numbers, special characters)
    tokens = [word for word in tokens if word.isalpha()]

    # 7. Stopwords Removal
    tokens = [word for word in tokens if word not in stop_words]

    # 8. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 9. Handling Repeated Characters (e.g., "loooove" -> "love")
    tokens = [re.sub(r'(.)\1{2,}', r'\1', word) for word in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Example usage
example_text = "Looove! Check out this amazing blog poooooost at https://example.com. It’s the best!!"
cleaned_text = preprocess_text(example_text)
print("Original Text:", example_text)
print("Preprocessed Text:", cleaned_text)
