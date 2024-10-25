
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Read the data

df = pd.read_csv("text_emb_data.csv")

def convert_embedding(embedding_str):
    embedding_list = embedding_str.strip('[]').split()
    return np.array(embedding_list, dtype=np.float32)

df['w2v_embedding'] = df['w2v_embedding'].apply(convert_embedding)


# print(df['w2v_embedding'].shape)

X = np.array(df['w2v_embedding'].tolist())  

# print(X.shape)



y = df['level']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# different svm kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[kernel] = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }

results_df = pd.DataFrame(results).T
print(results_df)



"""
accuracy  precision  recall  f1-score
linear     0.7575   0.757703  0.7575  0.757501
poly       0.7740   0.774256  0.7740  0.773997
rbf        0.7740   0.774409  0.7740  0.773981
sigmoid    0.6915   0.691686  0.6915  0.691502

"""



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Define the input and output shapes
input_dim = X.shape[1]
output_classes = len(np.unique(y))

# Convert labels to one-hot encoding if there are multiple classes
y = pd.get_dummies(y).values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Deep Averaging Network model
model = Sequential([
    Input(shape=(input_dim,)),             # Input layer with the shape of the embeddings
    Dense(256, activation='relu'),         # Hidden layer with ReLU activation
    Dropout(0.5),                          # Dropout for regularization
    Dense(128, activation='relu'),         # Another hidden layer with ReLU
    Dropout(0.5),                          # Another dropout layer
    Dense(output_classes, activation='softmax')  # Output layer with softmax activation for classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("Classification Report:\n", classification_report(y_true_labels, y_pred_labels))
print("Accuracy:", accuracy_score(y_true_labels, y_pred_labels))
