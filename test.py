from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def test_svm_classifier_accuracy():
    # Load dataset and split it
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Train SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Assert if accuracy is greater than 0
    accuracy = clf.score(X_test, y_test)
    assert accuracy > 0
    