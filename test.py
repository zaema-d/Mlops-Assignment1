import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Test if the SVM classifier can be trained and has an accuracy score
def test_svm_classifier_accuracy():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    assert accuracy > 0

# Test if the classifier can predict on a new sample
def test_svm_classifier_prediction():
    clf = joblib.load('breast_cancer_model.pkl')
    sample = [[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
           3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
           8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
           3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
           1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01]] 
    prediction = clf.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]  

