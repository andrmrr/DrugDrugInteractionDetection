#! /bin/bash

BASEDIR=../../../DDI
#DDIBASEDIR=/Users/danilakokin/Desktop/UPC/Semester2/MUD/lab_resources/DDI

# Define directory paths for training, prediction, and output
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DIR=$SCRIPT_DIR/train
PREDICT_DIR=$SCRIPT_DIR/predict
OUTPUT_DIR=$SCRIPT_DIR/output
MODELS_DIR=$SCRIPT_DIR/models

# Create these directories if they don't exist
mkdir -p "$TRAIN_DIR"
mkdir -p "$PREDICT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODELS_DIR"

#pip install -q -r requirements.txt
#
##UNCOMMENT THIS PART WHEN YOU CHANGE FEATURES
#./corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
#sleep 1
#
## extract features
#echo "Extracting features"
#python3 extract-features.py "$BASEDIR/data/test/" "$BASEDIR/resources/HSDB.txt" > "$TRAIN_DIR/test.cod" &
#python3 extract-features.py "$BASEDIR/data/train/" "$BASEDIR/resources/HSDB.txt" | tee "$TRAIN_DIR/train.cod" | cut -f4- > "$TRAIN_DIR/train.cod.cl"
#
#kill `cat /tmp/corenlp-server.running`


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 "$TRAIN_DIR/train-sklearn.py" "$MODELS_DIR/model_nb.joblib" "$PREDICT_DIR/vectorizer.joblib" < "$TRAIN_DIR/train.cod.cl"
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 "$PREDICT_DIR/predict-sklearn.py" "$MODELS_DIR/model_nb.joblib" "$PREDICT_DIR/vectorizer.joblib" < "$TRAIN_DIR/test.cod" > "$OUTPUT_DIR/test-NB.out"
# evaluate Naive Bayes results
echo "Evaluating Naive Bayes results..."
python3 evaluator.py DDI "$BASEDIR/data/test" "$OUTPUT_DIR/test-NB.out" > "$OUTPUT_DIR/test-NB.stats"

# Best parameters found:  {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
#-------- Best accuracy achieved:  0.35917328841637786

# train XGBoost model
#echo "Training XGBoost model..."
#python3 "$TRAIN_DIR/train-xgb.py" "$MODELS_DIR/model_xgb.joblib" "$PREDICT_DIR/vectorizer.joblib" "$PREDICT_DIR/label_encoder_xgb.joblib" < "$TRAIN_DIR/train.cod.cl"
## run XGBoost model
#echo "Running XGBoost model..."
#python3 "$PREDICT_DIR/predict-xgb.py" "$MODELS_DIR/model_xgb.joblib" "$PREDICT_DIR/vectorizer.joblib" "$PREDICT_DIR/label_encoder_xgb.joblib" < "$TRAIN_DIR/test.cod" > "$OUTPUT_DIR/test-XGB.out"
## evaluate XGBoost results
#echo "Evaluating XGBoost results..."
#python3 evaluator.py DDI "$BASEDIR/data/test" "$OUTPUT_DIR/test-XGB.out" > "$OUTPUT_DIR/test-XGB.stats"

# train SVC model
echo "Training SVC model..."
python3 "$TRAIN_DIR/train-svc.py" "$MODELS_DIR/model_svc.joblib" "$PREDICT_DIR/vectorizer.joblib" < "$TRAIN_DIR/train.cod.cl"
# run SVC model
echo "Running SVC model..."
python3 "$PREDICT_DIR/predict-sklearn.py" "$MODELS_DIR/model_svc.joblib" "$PREDICT_DIR/vectorizer.joblib" < "$TRAIN_DIR/test.cod" > "$OUTPUT_DIR/test-SVC.out"
# evaluate SVC results
echo "Evaluating SVC results..."
python3 evaluator.py DDI "$BASEDIR/data/test" "$OUTPUT_DIR/test-SVC.out" > "$OUTPUT_DIR/test-SVC.stats"
