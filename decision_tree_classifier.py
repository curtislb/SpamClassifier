from classification import train_classifier, test_classifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    classifier = DecisionTreeClassifier()
    train_classifier(classifier)
    test_classifier(classifier)
