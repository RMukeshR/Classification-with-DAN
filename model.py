# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # Load the dataset
# final_df = pd.read_csv('combined_embeddings_dataset.csv')



# # Convert space-separated strings to lists of floats
# def convert_embeddings(embedding_str):
#     # Remove brackets and split by space, then convert to float
#     return np.array([float(x) for x in embedding_str.strip('[]').split()])

# # Apply the conversion function
# final_df['word2vec_embeddings'] = final_df['word2vec_embeddings'].apply(convert_embeddings)

# # Select the Word2Vec embeddings and labels
# X = np.array(final_df['word2vec_embeddings'].tolist())
# y = final_df['label'].map({'positive': 1, 'negative': 0})  # Convert labels to binary

# # Split the dataset into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Initialize classifiers
# classifiers = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Random Forest': RandomForestClassifier(),
#     'Support Vector Machine': SVC(),
# }

# # Train and evaluate each classifier
# for name, clf in classifiers.items():
#     # Fit the model
#     clf.fit(X_train, y_train)
    
#     # Predict on training and validation data
#     y_train_pred = clf.predict(X_train)
#     y_val_pred = clf.predict(X_val)
    
#     # Calculate accuracy
#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     val_accuracy = accuracy_score(y_val, y_val_pred)
    
#     print(f"{name}:")
#     print(f"  Training Accuracy: {train_accuracy:.4f}")
#     print(f"  Validation Accuracy: {val_accuracy:.4f}")






import pandas as pd

df = pd.read_csv("text_emb_data.csv")

print(df.head())