from bow import *
from random import shuffle
from nltk.classify.scikitlearn import SklearnClassifier
import nltk.classify
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import tree
import os
import sys
from sklearn.svm import LinearSVC


total_vocabulary = {}
children_lit = []
advanced_lit = []
all_lit = []
fp1 = "directories_of_children_lit"
fp2 = "directories_of_advanced_lit"

# dictionaries are counts of how many times each word appears in text. This
# method turn those raw counts into percentages of what percentage of times each words s in text
# f.e. {I: 5, eat: 4, pizza: 3} beocmes {I: 5/12, eat: 1/3, pizza: 1/4}

"""def convert_raw_frequency_vectors(dictionary):
	total_words = 0
	for word in dictionary:
		total_words += dictionary[word]
	new_dictionary = {word: dictionary[word]/total_words} for word in dictionary
	return new_dictionary"""


for directory in os.listdir(fp1):
	subdir = "directories_of_children_lit/" + directory
	for string in os.listdir(subdir):
		file_name = "directories_of_children_lit/" + directory + '/' + string
		words = Document()
		words.read_files(file_name)
		dictionary = dict(words)
		children_lit.append(dictionary)
		#all_lit.append(dictionary)
		#for word in dictionary:
		#	total_vocabulary[word] = True
print("finished first part")

for directory in os.listdir(fp2):
	subdir = "directories_of_advanced_lit/" + directory
	for string in os.listdir(subdir):
		file_name = "directories_of_advanced_lit/" + directory + '/' + string
		words = Document()
		words.read_files(file_name)
		dictionary = dict(words)
		advanced_lit.append(dictionary)
		#all_lit.append(dictionary)
		#for word in dictionary:
		#	total_vocabulary[word] = True

print("finished second part")

training_data = []
test_data_1 = []
test_data_2 = []

test_fp1 = "testing_material/children_training/"
test_fp2 = "testing_material/advanced_training/"

for dictionary in children_lit:
	training_data.append((dictionary, "children"))
for dictionary2 in advanced_lit:
	training_data.append((dictionary2, "advanced"))
shuffle(training_data)
classif = None
if sys.argv[1] == "svm":
	print("doing svm")
	classif = SklearnClassifier(LinearSVC()).train(training_data)
elif sys.argv[1] == "tree":
	print("doing tree")
	classif = SklearnClassifier(tree.DecisionTreeClassifier()).train(training_data)
else:
	classif = SklearnClassifier(BernoulliNB()).train(training_data)
	print("doing Bernoulli Naive Bayes")
#classif = nltk.classify.NaiveBayesClassifier.train(training_data)
#test_data = [{"That": 1, "one": 1, "!": 1, "You": 1, "will":1, "do": 1, "it":1}]

for directory in os.listdir(test_fp1):
	subdir = test_fp1 + directory
	if directory.startswith("."):
		continue
	for string in os.listdir(subdir):
		file_name = test_fp1 + directory + '/' + string
		words = Document()
		words.read_files(file_name)
		dictionary = dict(words)
		test_data_1.append(dictionary)

for directory in os.listdir(test_fp2):
	subdir = test_fp2 + directory
	if directory.startswith("."):
		continue
	for string in os.listdir(subdir):
		file_name = test_fp2 + directory + '/' + string
		words = Document()
		words.read_files(file_name)
		dictionary = dict(words)
		test_data_2.append(dictionary)

results_1 = classif.classify_many(test_data_1)
results_2 = classif.classify_many(test_data_2)
total_1 = len(results_1)
total_2 = len(results_2)
correct_1 = 0
correct_2 = 0
for string in results_1:
	if "children" in string:
		correct_1 += 1
for string in results_2:
	if "advanced" in string:
		correct_2 += 1

print("For children text: %i correct out of %i for a percentage of %f" % (correct_1, total_1, correct_1/total_1))
print("For advanced text: %i correct out of %i for a percentage of %f" % (correct_2, total_2, correct_2/total_2))
print("total classification percentage is %f" % ((correct_1 + correct_2)/(total_1 + total_2)))

"""probs = classif.prob_classify_many(test_data)
for item in probs:
	print('%.4f %.4f' % (item.prob("children"), item.prob("advanced")))"""
