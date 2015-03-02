from classification import train_classifier, test_classifier, cross_validation
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    classifier = RandomForestClassifier()
    train_classifier(classifier)
    test_classifier(classifier)
    print '------------------'
    cross_validation(classifier)
