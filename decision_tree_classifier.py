from email_process import read_bagofwords_dat, read_classes_txt
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

# Training Phase ##############################################################

    filename = 'trec07p_data/Train/train_emails_bag_of_words_200.dat'
    words = read_bagofwords_dat(filename, 45000)
    
    classes = read_classes_txt('trec07p_data/Train/train_emails_classes_200.txt')
    
    classifier = DecisionTreeClassifier()
    classifier.fit(words, classes)

# Testing Phase ###############################################################

    filename = 'trec07p_data/Test/test_emails_bag_of_words_0.dat'
    words = read_bagofwords_dat(filename, 5000)
    predictions = classifier.predict(words)
    
    classes = read_classes_txt('trec07p_data/Test/test_emails_classes_0.txt')
    
    num_correct = 0
    for i in range(len(predictions)):
        if classes[i] == predictions[i]:
            num_correct += 1
    
    print (float(num_correct) / float(len(predictions)))
