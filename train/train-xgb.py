import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from joblib import dump, load
from train_utils import load_data


if __name__ == '__main__':
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]
    label_encoder_file = sys.argv[3]

    train_features, y_train = load_data(sys.stdin)
    y_train = np.asarray(y_train)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    v = load(vectorizer_file)
    X_train = v.transform(train_features)

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.3, 0.5, 0.7]
    }

    clf = XGBClassifier(eval_metric='mlogloss')
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro', verbose=0)
    grid_search.fit(X_train, y_train_encoded)

    print("-------- Best parameters found: ", grid_search.best_params_)
    print("-------- Best accuracy achieved: ", grid_search.best_score_)

    # Save the best model and label encoder
    dump(grid_search.best_estimator_, model_file)
    dump(label_encoder, label_encoder_file)
