# Matthew Scott, April 5, 2022
# This program is a word sense disambiguation training model.
# A word 'line' can have two contexts, or senses, either "product" , or "phone"
# the program will go through manually marked training data, with the correct sense for every sentence containing line
# it will create probability scores using every word in the sentence so that it can predict future contexts (senses)

# Then it will go through the testing data, that does not contain senses
# it will go through each sentence, and using the previously created discrimination scores, try and predict
#   which sense should be applied to each sentence. correctly identifying the meaning of the word "line",
#   for that sentence.

# To use this program, type into the terminal
# python wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
# the program will output a my-model.txt file with information about what the model produced, for example:
# The instance ID, the sense for that instance, and the discriminative word and its log score.
# ('line-n.w8_059:8174:', 'Phone', 'telephone', 6.729649542218809)
# It will also output the line-answers, the output of the correct instance id and sense
# to be compared with the correctly tagged key file.

# Example input/output for scorer.py
# input:
# python scorer.py my-line-answers.txt line-key.txt
# output example: This will be a confusion matrix that will show you the actual word sense, and the predicted.
# you will also see an accuracy score

# Predicted  Phone  Product
# Actual
# Phone         69        3
# Product       16       38
# Accuracy score: 84.92063492063492 %
# Most frequent sense baseline: 0.5714285714285714%

# The most frequent sense baseline for the correct key is 'Phone', containing 72 out of 126 total instances (57%)

import re
import math
import sys

line_train = open(sys.argv[1], 'r').read().split('\n')
line_test = open(sys.argv[2], 'r').read().split('\n')
model = open(sys.argv[3], "x")

# list of stopwords written out, courtesy of nltk stopwords, manually included the words 'line' and 'lines'
# from nltk.corpus import stopwords
# import nltk
# nltk.download("stopwords")
# stop_words = stopwords.words("english")
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
              'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
              'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
              'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
              "won't", 'wouldn', "wouldn't", "line", "lines"]

# This is the pre-processing step that I did with the training data.
# The data is split into the sense, and context.
# The context sentences, are stripped of markup, punctuation, and stops words. Then tokenized into a list by each word.
# the train_data will be in the format: train_data[0][1], where [Product, [context sentence]]
# Every word in the training data, except line or lines, will be used as a feature. bag of words model.
train_data = []
sense = "Undefined"
for instances in line_train:
    if 'senseid="phone"' in instances:
        sense = "Phone"
    if 'senseid="product"' in instances:
        sense = "Product"
    if '<s>' in instances:
        removed_tags = re.sub(r'<[^>]*>', '', instances)  # Remove markup tags
        removed_punctuation = re.findall(r"[\w+']+", removed_tags)  # Remove punctuation, returns a list
        stop = []
        for words in removed_punctuation:  # Iterating over formatted list to create new list without stop words.
            if words not in stop_words:
                stop.append(words)
        train_data.append([sense, stop])  # Final formatted training data

word_sense_frequency = {  # The word sense frequency will be a nested dictionary.
    "Phone": {
    },
    "Product": {
    }
}
word_discrimination_score = {}  # This list contains each word in the training data, with a number associated with it.
sense_frequency = {}  # How often each sense is seen in the training data.
# This loop will find the sense frequencies, and the word/sense frequencies
for items in train_data:
    if items[0] not in sense_frequency:  # items[0] are the senses
        sense_frequency[items[0]] = 1
    else:
        sense_frequency[items[0]] += 1

    for word in items[1]:  # items[1] are the list of words from the context
        if word not in word_sense_frequency[items[0]]:
            word_sense_frequency[items[0]].update({word: 1})
        else:
            word_sense_frequency[items[0]][word] += 1

# This loop will go through each word in the training data, and give a discrimination score to each.
# For any word that is in one sense but not the other, a default value of '0.1' will be assigned to it.
# The key: value will be in word_discrimination_score dictionary.
for items in train_data:
    for word in items[1]:
        word_discrimination_score[word] = abs(
            math.log((word_sense_frequency["Phone"].get(word, 0.1) / sense_frequency["Phone"]) /
                     (word_sense_frequency["Product"].get(word, 0.1) / sense_frequency["Product"])))

# This is the same preprocessing step as in the train_data.
# A list is made where the first item in the list is an instance id, and the second item is a list of words.
# ['instance_id', ['list', 'of', 'tokenized' 'words']]
# The instance id is found from a regex, and the sentences are stripped of markup, punctuation, and stop words.
test_data = []
instance_id = "Undefined"
for instances in line_test:
    if "<instance id=" in instances:
        instance_id = re.findall(r'\"(.*?)\"', instances)[0]
    if '<s>' in instances:
        removed_tags = re.sub(r'<[^>]*>', '', instances)
        removed_punctuation = re.findall(r"[\w+']+", removed_tags)
        stop = []
        for words in removed_punctuation:
            if words not in stop_words:
                stop.append(words)
        test_data.append([instance_id, stop])

# This loop will go through each word in the test data sentences
# It will find the discrimination scores from the training data and remember it for each word, and find the highest
# The word with the highest score (max_word) will be used to look up the related sense from the training data
for items in test_data:
    max_word = ""  # This will be the word with the highest discrimination score in each sentence, per instance.
    max_word_score = 0  # This will be  the actual value associated with that word
    for word in items[1]:
        # for each word in test data, get the word with the highest score from the training data
        if word_discrimination_score.get(word, 0) > max_word_score:
            max_word_score = word_discrimination_score.get(word, 0)
            max_word = word
    # For the word that had the highest value in the test data, find the related sense in the training data.
    sense_phone = word_sense_frequency["Phone"].get(max_word, 0)
    sense_product = word_sense_frequency["Product"].get(max_word, 0)
    if sense_phone > sense_product:
        result = "Phone"
    else:
        result = "Product"
    # instance id, sense, the determining word, the log score of that word
    write = items[0], result, max_word, max_word_score
    model.write(str(write) + "\n")
    print(items[0], result)
