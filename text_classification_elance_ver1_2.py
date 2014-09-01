from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
from collections import Counter
from numpy import genfromtxt, savetxt
import csv
import os
import numpy as np
import re
import sys
import unicodedata

# Function to clean up the data - start
def clean_string(string):
    string = string.replace("'", "")  # Getting data clean - 1: Stripping apostrophe's
    string = string.replace("?", "")  # Getting data clean - 2: Stripping ?
    string = string.replace("!", "")  # Getting data clean - 3: Stripping !
    string = string.replace("\r\n", " ")  # Getting data clean - 4: Stripping return
    string = re.sub('[^\040-\176]', ' ', string) # Getting data clean - 5: Stripping unknown blank spaces and characters
    string = string.replace(",", "")  # Getting data clean - 6: Stripping comma
    
    # Getting data clean - 7: Stripping unicode characters
    string = re.sub(regex,"",string) 
    
    # Getting data clean - 8: Stripping multiple occurences/repeatations of same character to 1
    string = re.sub(r'([a-z])\1+', r'\1', string)
    
    return string
# Function to clean up the data - end

# Function to write file - start
def write_to_file(filename, final_prediction):
    savetxt(('data_twitter_life_event_classification/' + filename), final_prediction, delimiter=',', fmt='%s,%s', 
            header='tweet,category', comments = '')
    return
# Function to write file - end

# Main code - start
print "Data Processing Started..."

# Reading different datasets

#life_event = open('data_twitter_life_event_classification/wedding-and-engagement-manual-examples.csv') 
#life_event = open('data_twitter_life_event_classification/events-manual-examples-1.csv') 
life_event = open('data_twitter_life_event_classification/all-tagged-samples.csv') 

life_event_data = []
life_event_labels = []
csv_reader = csv.reader(life_event)
regex = r'[^\x00-\x7F]' # Regex to strip unicode characters

for line in csv_reader:
    #string = line[0]
    line[0] = clean_string(line[0])
    #line[0] = string
    life_event_data.append(line[0])
    life_event_labels.append((line[1]))

life_event.close()

# Reading irrelevant data to populate third category that is "irrelevant"

#not_life_event = open('data_twitter_life_event_classification/manual-irrelevant.csv') 
#not_life_event = open('data_twitter_life_event_classification/tagged-irrelevant-1.csv') 
not_life_event = open('data_twitter_life_event_classification/all-irrelevant-tagged-samples.csv') 

not_life_event_data = []
not_life_event_labels = []
csv_reader_not_life_event = csv.reader(not_life_event)


for line in csv_reader_not_life_event:
    #string = line[0]
    line[0] = clean_string(line[0])
    #line[0] = string
    life_event_labels.append((line[1])) # For latest data set named "all-irrelevant-tagged-samples"
    # For tagging all irrelevant samples as irrelevant and for wedding and engagement only dataset
    #not_life_event_labels.append('irrelevant')
    not_life_event_data.append(line[0])

not_life_event.close()

# Join the data lists together

life_event_data = life_event_data + not_life_event_data
life_event_labels = life_event_labels + not_life_event_labels

# Reading test data 

#test_data_wed_eng = open('data_twitter_life_event_classification/untagged-samples.csv') 
test_data_wed_eng = open('data_twitter_life_event_classification/all-untagged.csv')

test_data_wed_eng_data = []
#life_event_labels = []
csv_reader = csv.reader(test_data_wed_eng)
regex = r'[^\x00-\x7F]' # Regex to strip unicode characters

for line in csv_reader:
    #string = line[1] # Uncomment for file untagged-samples.csv
    string = line[0] # Uncomment for file all-untagged.csv
    string = clean_string(string)
    #line[1] = string # Uncomment for file untagged-samples.csv
    line[0] = string #Uncomment for file all-untagged.csv
    #life_event_labels.append((line[1]))
    #test_data_wed_eng_data.append(line[1]) # Uncomment for file untagged-samples.csv
    test_data_wed_eng_data.append(line[0]) #Uncomment for file all-untagged.csv

test_data_wed_eng.close()

#Getting 70% of the dataset as training - Rest 30% would be used as test for cross validation

# Equally splitting the data according to the ratio of categories present. This method preserves the ratio

X_train, X_test, y_train, y_test = train_test_split(life_event_data, life_event_labels,
    test_size=0.30, random_state=123)

# Uncomment this line for applying model to test data 

X_test = test_data_wed_eng_data

# A variation of options used to generate Tfidf Vector

#vectorizer = TfidfVectorizer(min_df=0, 
#    ngram_range=(1, 3), 
#    stop_words='english', 
#    strip_accents='unicode', 
#    norm='l2')

vectorizer = TfidfVectorizer(min_df=0, 
    ngram_range=(1, 5), 
    stop_words='english', 
    strip_accents='unicode', 
    norm='l2')

test_string = unicode(life_event_data[1])

print "Example string: " + test_string
print "Preprocessed string: " + vectorizer.build_preprocessor()(test_string)
print "Tokenized string:" + str(vectorizer.build_tokenizer()(test_string))
print "N-gram data string:" + str(vectorizer.build_analyzer()(test_string))
print "\n"

print "Length of Training Data Used: {:}".format(len(X_train)) 
print "Length of Test Data: {:}".format(len(X_test)) 
print "Length of Total Training Data: {:}".format(len(life_event_data)) 
print "\n"
print "Data Processing Complete..."

print "Prediction Started..."

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Code for prediction of test set - To be uncommented when cross validation is not in use.
'''
# Need to comment this - start
nb_classifier = MultinomialNB().fit(X_train, y_train)

y_nb_predicted = nb_classifier.predict(X_test)
# Need to comment this - end
'''

# Applying labels to untagged data - start
nb_classifier = MultinomialNB().fit(X_train, y_train)

y_nb_predicted = nb_classifier.predict(X_test)
#print y_nb_predicted
# # Applying labels to untagged data - end

print "Prediction Complete..."

'''
# Simple Cross validation for the data - Works fine - Results are posted.

#scores = cross_validation.cross_val_score( nb_classifier, X_train, y_train, cv=10) # This method is working and returning score

# X_train was a sparse matrix. It needed to be converted to numpy array to be used in cross validation. 
# X_train is not changed itself but a copy is made because it was used by prediction of test data

X_train_numpy = X_train.todense()
X_train_numpy = np.array(X_train_numpy)
accuracy_k_fold_cv = []
k_folds = 2  # Change the value of k to see number of folds cross validation accuracy 

# K-Fold Cross Validation for NB code starts here

cv = cross_validation.KFold(X_train.shape[0], n_folds = k_folds, indices = False)

for cv_index, (train, test) in enumerate(cv):
    # Uncomment this code to see each fold's accuracy score. - Part 1
    
    # Need to comment this - start
#    print("# Cross Validation Iteration #%d" % cv_index)
#    print("train indices: {0}...".format(train[:10]))
#    print("test indices: {0}...".format(test[:10]))
    # Need to comment this - end
    
    X = X_train_numpy[:,train]
    Y = y_train[train]
    nb_classifier_cv = MultinomialNB().fit(X_train_numpy[train], y_train[train])
    y_nb_predicted_cv = nb_classifier_cv.predict(X_train_numpy[test])
    accuracy_k_fold_cv.append(metrics.accuracy_score(y_train[test], y_nb_predicted_cv))
    
    # Uncomment this code to see each fold's accuracy score. - Part 2
    #print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_train[test], y_nb_predicted_cv))

print accuracy_k_fold_cv
print 'The average accuracy for this classifier is ' + str(sum(accuracy_k_fold_cv)/float(k_folds))


# K-Fold Cross Validation code ends here
'''
'''

# Code to calculate accuracy and rest of the statistics for test data 
# Commented to test cross validation accuracy
# Need to comment this - start
print "MODEL: Multinomial Naive Bayes\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_nb_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_nb_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_nb_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_nb_predicted))

print '\nHere is the classification report:'
print classification_report(y_test, y_nb_predicted)

#simple thing to do would be to up the n-grams to bigrams; try varying ngram_range from (1, 1) to (1, 2)
#we could also modify the vectorizer to stem or lemmatize
print '\nHere is the confusion matrix:'
#print metrics.confusion_matrix(y_test, y_nb_predicted, labels=unique(nyt_labels))
print metrics.confusion_matrix(y_test, y_nb_predicted, labels = np.unique(life_event_labels))
# Need to comment this - end
'''

print "Printing File Start..."

# Writing Results to different files according to the training data

final_prediction = zip(test_data_wed_eng_data, y_nb_predicted)

#write_to_file('untagged-samples-experiment3.csv', final_prediction)
#write_to_file('untagged-samples-experiment2.csv', final_prediction)

# File created for Data learnt from manual-irrelevant.csv and wedding-and-engagement-manual-examples.csv
#write_to_file('untagged-samples-experiment0.csv', final_prediction)

# I am calling Data learnt from manual-irrelevant.csv and wedding-and-engagement-manual-examples.csv as Experiment 0
#write_to_file('all-untagged-experiment0.csv', final_prediction)

#write_to_file('all-untagged-experiment3.csv', final_prediction)
write_to_file('all-untagged-experiment4.csv', final_prediction) # Testing file

print "Printing File Complete..."

# Main code - end

