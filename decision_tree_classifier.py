from classification import train_classifier, test_classifier, cross_validation
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    classifier = DecisionTreeClassifier()
    train_classifier(classifier)
    test_classifier(classifier)
    print '------------------'
    cross_validation(classifier)
