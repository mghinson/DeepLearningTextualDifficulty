from gensim.models.doc2vec import *
import pandas as pd
import numpy as np
import os
import sys

number = int(sys.argv[1])

def score(documents, dict1, n):
	the_score = 0
	for doc in documents:
		tag = doc[0]
		#print(tag)
		if tag in dict1:
			the_score += 1
	return the_score/n

def is_classified_correct(model, isChildren, file_path):
	tag = model.docvecs[file_path]
	sims = model.docvecs.most_similar(file_path, topn=number)
	if isChildren:
		the_score = score(sims, children_dictionary, number)
		if the_score > .5:
			return True
		else:
			return False
	else:
		the_score = score(sims, advanced_dictionary, number)
		if the_score > .5:
			return True
			#print("Categorization of %s is as 'more advanced' text" % title)
		else:
			return False
			#print("Categorization of %s is as 'children' text" % title)

# the following method is taken from Matt Taddy's additions to the Gensim package

def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
    # now exponentiate to get likelihoods, 
    #lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    lhd = np.exp(llhd)
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob

documents = []
children_docs = []
advanced_docs = []
test_children_docs = []
test_advanced_docs = []
i = 1
fp_1 = "directories_of_children_lit/"
fp_2 = "directories_of_advanced_lit/"
test_fp1 = "testing_material/children_training/"
test_fp2 = "testing_material/advanced_training/"

all_files = {}
children_dictionary = {}
advanced_dictionary = {}

for directory in os.listdir(fp_1):
	subdir = fp_1 + directory
	if directory.startswith("."):
		continue
	for a_file in os.listdir(subdir):
		filename = subdir + '/' + a_file
		tagged = TaggedDocument(filename, tags=["children"])
		tagged2 = TaggedDocument(filename, tags=[filename])
		documents.append(tagged2)
		children_docs.append(tagged)
		all_files[filename] = filename
		children_dictionary[filename] = i
		#children_dictionary[i] = filename
		i += 1

for directory in os.listdir(test_fp1):
	subdir = test_fp1 + directory
	if directory.startswith("."):
		continue
	for a_file in os.listdir(subdir):
		filename = subdir + '/' + a_file
		tagged = TaggedDocument(filename, tags=["children"])
		tagged2 = TaggedDocument(filename, tags=[filename])
		documents.append(tagged2)
		test_children_docs.append(filename)
		all_files[filename] = filename
		children_dictionary[filename] = i
		#children_dictionary[i] = filename
		i += 1

for directory in os.listdir(fp_2):
	subdir = fp_2 + directory
	if directory.startswith("."):
		continue
	for a_file in os.listdir(subdir):
		filename = subdir + '/' + a_file
		tagged = TaggedDocument(filename, tags=["advanced"])
		tagged2 = TaggedDocument(filename, tags=[filename])
		documents.append(tagged2)
		advanced_docs.append(tagged)
		all_files[filename] = filename
		advanced_dictionary[filename] = i
		#advanced_dictionary[i] = filename
		i += 1
for directory in os.listdir(test_fp2):
	subdir = test_fp2 + directory
	if directory.startswith("."):
		continue
	for a_file in os.listdir(subdir):
		filename = subdir + '/' + a_file
		tagged = TaggedDocument(filename, tags=["advanced"])
		tagged2 = TaggedDocument(filename, tags=[filename])
		documents.append(tagged2)
		test_advanced_docs.append(filename)
		all_files[filename] = filename
		advanced_dictionary[filename] = i
		#advanced_dictionary[i] = filename
		i += 1


print("starting model")
advanced_docs = advanced_docs[0:5834]

# approach: "Nearest Neighbor-ish search" - used gensim's n most likely function to compute most similar
# documents across all documents

model = Doc2Vec(documents, size=100, min_count=5, alpha=.025)
#model.save("doc2vec1.model")
total_1 = 0
correct_1 = 0
total_2 = 0
correct_2 = 0
for directory in os.listdir(test_fp1):
	subdir = test_fp1 + directory
	if directory.startswith("."):
		continue
	for a_file in os.listdir(subdir):
		filename = subdir + '/' + a_file
		if is_classified_correct(model, True, filename):
			correct_1 += 1
		total_1 += 1
print("%i classified correct out of %i for children's literature" % (correct_1, total_1))

for directory in os.listdir(test_fp2):
	subdir = test_fp2 + directory
	if directory.startswith("."):
		continue
	for a_file in os.listdir(subdir):
		filename = subdir + '/' + a_file
		if is_classified_correct(model, False, filename):
			correct_2 += 1
		total_2 += 1
print("%i classified correct out of %i for advanced literature" % (correct_2, total_2))


# Part 2 - NN training w/out of the box classification

children_model = Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)
advanced_model = Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)

children_model.build_vocab(children_docs)
print(len(children_docs))
print(len(advanced_docs))
for epoch in range(15):
    children_model.train(children_docs)
    children_model.alpha -= 0.007 # decrease the learning rate
    children_model.min_alpha = children_model.alpha # fix the learning rate, no deca
    children_model.train(children_docs)

advanced_model.build_vocab(advanced_docs)
for epoch in range(15):
    advanced_model.train(advanced_docs)
    advanced_model.alpha -= 0.007 # decrease the learning rate
    advanced_model.min_alpha = children_model.alpha # fix the learning rate, no deca
    advanced_model.train(advanced_docs)

new_file = open('children_results.txt', 'w+')
file_2 = open('advanced_results.txt', 'w+')
mods = [children_model, advanced_model]
child_results = docprob(test_children_docs, mods)
advanced_results = docprob(test_advanced_docs, mods)
for index, row in child_results.iterrows():
	new_file.write(str(row))
	new_file.write('\n')

for index, row in advanced_results.iterrows():
	file_2.write(str(row))
	file_2.write('\n')

new_file.close()
file_2.close()






#print(str(child_results))





