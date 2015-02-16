
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


def get_files(mypath):
    return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
    return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def tokenize_corpus(path, train=True):
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    classes = []
    samples = []
    docs = []
    dirs = get_dirs(path)
    if train == True:
        words = {}
    for dir in dirs:
        files = get_files(path+"/"+dir)
        for f in files:
            classes.append(dir)
            samples.append(f)
            inf = open(path+'/'+dir+'/'+f,'r')
            raw = inf.read().decode('latin1') # or ascii or utf8 or utf16
            # remove noisy characters; tokenize
            raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
            tokens = word_tokenize(raw)
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            tokens = [w for w in tokens if w not in stopWords]
            tokens = [wnl.lemmatize(t) for t in tokens]
            tokens = [porter.stem(t) for t in tokens]
            if train == True:
                for t in tokens: 
                    # this is a hack but much faster than lookup each
                    # word within many dict keys
                    try:
                        words[t] = words[t]+1
                    except:
                        words[t] = 1
            docs.append(tokens)
    if train == True:
        return(docs, classes, samples, words)
    else:
        return(docs, classes, samples)
        

def wordcount_filter(words, num=5):
    keepset = []
    for k in words.keys():
        if(words[k] > num):
            keepset.append(k)
    print len(keepset)
    return(sorted(set(keepset)))


def find_wordcounts(docs, vocab):
    bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
    vocabIndex={}
    for i in range(len(vocab)):
       vocabIndex[vocab[i]]=i

    for i in range(len(docs)):
        doc = docs[i]

        for t in doc:
           index_t=vocabIndex.get(t)
           if index_t>=0:
              bagofwords[i,index_t]=bagofwords[i,index_t]+1

    print "Finished find_wordcounts for : "+str(len(docs))+"  docs"
    return(bagofwords)


# path should have one folder for each class. Class folders should
# contain text documents that are labeled with the class label (folder
# name). Bag of words representation, vocabulary will be output to
# <outputfile>_*.dat files.
def main(argv):
   path = ''
   outputf = ''
   vocabf = ''
   start_time = time.time()

   try:
      opts, args = getopt.getopt(argv,"p:o:v:",["path=","ofile=","vocabfile="])
   except getopt.GetoptError:
      print 'python text_process.py -p <path> -o <outputfile> -v <vocabulary>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'text_process.py -p <path> -o <outputfile> -v <vocabulary>'
         sys.exit()
      elif opt in ("-p", "--path"):
         path = arg
      elif opt in ("-o", "--ofile"):
         outputf = arg
      elif opt in ("-v", "--vocabfile"):
         vocabf = arg
	 
   print 'Path is "', path
   print 'Output file name is "', outputf
   print 'vocabulary file is "', vocabf
   if (not vocabf):
      (docs, classes, samples, words) = tokenize_corpus(path, train=True)
      word_count_threshold = 200
      vocab = wordcount_filter(words, num=word_count_threshold)
   else:
      vocabfile = open(path+vocabf, 'r')
      vocab = [line.rstrip('\n') for line in vocabfile]
      vocabfile.close()
      (docs, classes, samples) = tokenize_corpus(path, train=False)

   bow = find_wordcounts(docs, vocab)
   #sum over docs to see any zero word counts, since that would stink.
   x = numpy.sum(bow, axis=1) 
   print "doc with smallest number of words in vocab has: "+str(min(x))
   # print out files
   if (vocabf):
      word_count_threshold = 0   
   else:
      #outfile= open(path+"/"+outputf+"_vocab_"+str(word_count_threshold)+".txt", 'w')
      outfile= codecs.open(path+"/"+outputf+"_vocab_"+str(word_count_threshold)+".txt", 'w',"utf-8-sig")
      outfile.write("\n".join(vocab))
      outfile.close()
   #write to binary file for large data set
   bow.tofile(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".dat")

   #write to text file for small data set
   #bow.tofile(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".txt", sep=",", format="%s")
   outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
   outfile.write("\n".join(classes))
   outfile.close()
   outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
   outfile.write("\n".join(samples))
   outfile.close()
   print str(time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])



