#!/usr/bin/env python3

import sys
from joblib import load
from predict_utils import prepare_instances


if __name__ == '__main__':
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]
    label_encoder_file = sys.argv[3]

    # Load the trained model, vectorizer, and label encoder
    model = load(model_file)
    v = load(vectorizer_file)
    label_encoder = load(label_encoder_file)

    for line in sys.stdin:

        fields = line.strip('\n').split("\t")
        (sid, e1, e2) = fields[0:3]
        vectors = v.transform(prepare_instances([fields[4:]]))
        print(prepare_instances([fields[4:]]))
        prediction = model.predict(vectors)

        if prediction != "null":
            decoded_prediction = label_encoder.inverse_transform(prediction)
            print(sid, e1, e2, decoded_prediction[0], sep="|")




