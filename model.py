import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
import statistics as s

feature_matrix = np.load('feature_matrix_UT_Distance.npy', allow_pickle=True)
labels_array = np.load('labels_list_UT_Distance.npy', allow_pickle=True)

min_length = min(len(seq) for seq in feature_matrix)
feature_matrix_padded = pad_sequences(feature_matrix, dtype='float32', padding='post', maxlen=int(min_length))

classifier = SVC()


cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

scores = cross_val_score(classifier, feature_matrix_padded, labels_array, cv=cv, scoring='accuracy')


print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))
print("Accuracy in percentage:",round(np.mean(scores)*100,2) ,"%")
