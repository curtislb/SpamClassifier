from classification import train_classifier, test_classifier, cross_validation
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    classifier = AdaBoostClassifier()
    train_classifier(classifier, select_features=True)
    test_classifier(classifier, select_features=True)
    print '------------------'
    cross_validation(classifier, select_features=True)
