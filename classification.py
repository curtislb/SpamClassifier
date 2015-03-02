from email_process import read_bagofwords_dat, read_classes_txt
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import time

TRAIN_DIR = 'trec07p_data/Train/'
TEST_DIR = 'trec07p_data/Test/'

###############################################################################

# Training

def _train_classifier(classifier, X, y):
    start = time.time()
    classifier.fit(X, y)
    print 'Time:', (time.time() - start)

def train_classifier(classifier):
    filename = TRAIN_DIR + 'train_emails_bag_of_words_200.dat'
    words = read_bagofwords_dat(filename, 45000)
    classes = read_classes_txt(TRAIN_DIR + 'train_emails_classes_200.txt')
    _train_classifier(classifier, words, classes)
    
###############################################################################

# Testing

def _test_classifier(classifier, X, y, make_graphs=True):
    predictions = classifier.predict(X)
    pred_probs = classifier.predict_proba(X)[:,1]
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in xrange(len(predictions)):
        if predictions[i] == 1:
            if y[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if y[i] == 1:
                false_neg += 1
            else:
                true_neg += 1
    
    accuracy = float(true_pos + true_neg) / float(len(predictions))
    precision = float(true_pos) / float(true_pos + false_pos)
    recall = float(true_pos) / float(true_pos + false_neg)
    print 'Accuracy:', accuracy
    print 'Precision:', precision
    print 'Recall:', recall
    print 'F1 Score:', (2 * (precision * recall) / (precision + recall))
    
    if make_graphs:
        fpr, tpr, __ = roc_curve(y, pred_probs)
        plt.plot(fpr, tpr, 'o-')
        plt.plot([0.0, 1.0], [0.0, 1.0], '--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig('roc_curve.png')
        plt.clf()
        
        prec, rec, __ = precision_recall_curve(y, pred_probs)
        plt.plot(rec, prec, 'o-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig('pr_curve.png')

def test_classifier(classifier):
    filename = TEST_DIR + 'test_emails_bag_of_words_0.dat'
    words = read_bagofwords_dat(filename, 5000)
    classes = read_classes_txt(TEST_DIR + 'test_emails_classes_0.txt')
    _test_classifier(classifier, words, classes)
    
###############################################################################

# Cross Validation

def cross_validation(classifier, num_folds=5):
    filename = TRAIN_DIR + 'train_emails_bag_of_words_200.dat'
    words = read_bagofwords_dat(filename, 45000)
    classes = read_classes_txt(TRAIN_DIR + 'train_emails_classes_200.txt')
    
    skf = StratifiedKFold(classes, num_folds)
    fold = 1
    for train_index, test_index in skf:
        print 'Fold #', fold
        X_train, X_test = words[train_index], words[test_index]
        y_train, y_test = classes[train_index], classes[test_index]
        _train_classifier(classifier, X_train, y_train)
        _test_classifier(classifier, X_test, y_test, False)
        print '------------------'
        fold += 1
