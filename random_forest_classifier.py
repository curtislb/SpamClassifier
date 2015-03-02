from classification import train_classifier, test_classifier, cross_validation
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    classifier = RandomForestClassifier()
    train_classifier(classifier, select_features=False)
    test_classifier(classifier, select_features=False)
    print '------------------'
    cross_validation(classifier, select_features=False)
