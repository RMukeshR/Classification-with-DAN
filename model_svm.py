
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
df['glove_embedding'] = df['glove_embedding'].apply(convert_embedding)
df['fasttest_embedding'] = df['fasttest_embedding'].apply(convert_embedding)


emb = ["w2v_embedding", "glove_embedding", "fasttest_embedding"]



for i in emb:


# print(df['w2v_embedding'].shape)

    X = np.array(df[i].tolist())  

    # print(X.shape)



    y = df['level']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

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
    print(20*"_"+i+20*"_")
    results_df = pd.DataFrame(results).T
    print(results_df)



# """
# ____________________w2v_embedding____________________
#          accuracy  precision  recall  f1-score
# linear      0.790   0.790066   0.790  0.789954
# poly        0.804   0.804387   0.804  0.803987
# rbf         0.796   0.796115   0.796  0.796007
# sigmoid     0.714   0.714264   0.714  0.713992
# ____________________glove_embedding____________________
#          accuracy  precision  recall  f1-score
# linear      0.788   0.788360   0.788  0.787864
# poly        0.800   0.804103   0.800  0.799131
# rbf         0.798   0.798683   0.798  0.797800
# sigmoid     0.684   0.689217   0.684  0.681191
# ____________________fasttest_embedding____________________
#          accuracy  precision  recall  f1-score
# linear      0.798   0.798300   0.798  0.797891
# poly        0.808   0.808000   0.808  0.808000
# rbf         0.802   0.802073   0.802  0.801956
# sigmoid     0.680   0.680135   0.680  0.679795
# """
