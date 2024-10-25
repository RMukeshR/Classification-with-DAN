
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