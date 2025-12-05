#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.svm import SVC
from joblib import dump, load
from train_utils import load_data

if __name__ == '__main__':
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]

    # Read data from standard input
    train_features, y_train = load_data(sys.stdin)
    y_train = np.asarray(y_train)

    # Initialize the DictVectorizer
    v = load(vectorizer_file)
    X_train = v.transform(train_features)

    # Initialize the SVM classifier
    clf = SVC(kernel='linear', C=0.1, probability=True)  # Linear kernel, adjust C and kernel as needed

    # Train the classifier
    clf.fit(X_train, y_train)

    # Save the trained classifier
    dump(clf, model_file)