from email_process import read_bagofwords_dat, read_classes_txt
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import time

TRAIN_DIR = 'trec07p_data/Train/'
TEST_DIR = 'trec07p_data/Test/'

TRAIN_WORDS = read_bagofwords_dat(TRAIN_DIR + 'train_emails_bag_of_words_200.dat', 45000)
TRAIN_CLASSES = read_classes_txt(TRAIN_DIR + 'train_emails_classes_200.txt')
TEST_WORDS = read_bagofwords_dat(TEST_DIR + 'test_emails_bag_of_words_0.dat', 5000)
TEST_CLASSES = read_classes_txt(TEST_DIR + 'test_emails_classes_0.txt')

F_SELECTOR = SelectPercentile(percentile=5)
F_SELECTOR.fit(TRAIN_WORDS, TRAIN_CLASSES)

###############################################################################

# Training

def _train_classifier(classifier, X, y, f_selector=None):
    if f_selector is not None:
        X = f_selector.transform(X)
    start = time.time()
    classifier.fit(X, y)
    print 'Time:', (time.time() - start)

def train_classifier(classifier, select_features=False):
    if select_features:
        _train_classifier(classifier, TRAIN_WORDS, TRAIN_CLASSES, F_SELECTOR)
    else:
        _train_classifier(classifier, TRAIN_WORDS, TRAIN_CLASSES)
    
###############################################################################

# Testing

def _test_classifier(classifier, X, y, f_selector=None, make_graphs=True,):
    if f_selector is not None:
        X = f_selector.transform(X)
    
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
#                 print 'FP:', i
                false_pos += 1
        else:
            if y[i] == 1:
#                 print 'FN:', i
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
        plt.clf()
        plt.plot(fpr, tpr, 'o-')
        plt.plot([0.0, 1.0], [0.0, 1.0], '--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig('roc_curve.png')
        
        prec, rec, __ = precision_recall_curve(y, pred_probs)
        plt.clf()
        plt.plot(rec, prec, 'o-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig('pr_curve.png')

def test_classifier(classifier, select_features=False):
    if select_features:
        _test_classifier(classifier, TEST_WORDS, TEST_CLASSES, F_SELECTOR)
    else:
        _test_classifier(classifier, TEST_WORDS, TEST_CLASSES)
    
###############################################################################

# Cross Validation

def cross_validation(classifier, num_folds=5, select_features=False):
    skf = StratifiedKFold(TRAIN_CLASSES, num_folds)
    fold = 1
    for train_index, test_index in skf:
        print 'Fold #', fold
        X_train, X_test = TRAIN_WORDS[train_index], TRAIN_WORDS[test_index]
        y_train, y_test = TRAIN_CLASSES[train_index], TRAIN_CLASSES[test_index]
        if select_features:
            _train_classifier(classifier, X_train, y_train, F_SELECTOR)
            _test_classifier(classifier, X_test, y_test, F_SELECTOR, False)
        else:
            _train_classifier(classifier, X_train, y_train)
            _test_classifier(classifier, X_test, y_test, make_graphs=False)
        print '------------------'
        fold += 1

###############################################################################

if __name__ == '__main__':
    skb = SelectKBest(k=20)
    skb.fit(TRAIN_WORDS, TRAIN_CLASSES)
    feature_mask = skb.get_support()
    for i in xrange(len(feature_mask)):
        if feature_mask[i]:
            print i
    for score in skb.scores_[feature_mask]:
        print score
