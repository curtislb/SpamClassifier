from email_process import read_bagofwords_dat, read_classes_txt
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy

def train_classifier(classifier):
    filename = 'trec07p_data/Train/train_emails_bag_of_words_200.dat'
    words = read_bagofwords_dat(filename, 45000)
    classes = read_classes_txt('trec07p_data/Train/train_emails_classes_200.txt')
    classifier.fit(words, classes)

def test_classifier(classifier):
    filename = 'trec07p_data/Test/test_emails_bag_of_words_0.dat'
    words = read_bagofwords_dat(filename, 5000)
    predictions = classifier.predict(words)
    
    classes = read_classes_txt('trec07p_data/Test/test_emails_classes_0.txt')
    num_correct = 0
    for i in xrange(len(predictions)):
        if predictions[i] == classes[i]:
            num_correct += 1
    
    print 'Accuracy:', (float(num_correct) / float(len(predictions)))
    
    fpr, tpr, _ = roc_curve(classes, predictions)
    plt.plot(fpr, tpr, '.-')
    x = [0.01 * n for n in xrange(101)]
    plt.plot(x, x, '--')
    plt.savefig('roc_curve.png')
