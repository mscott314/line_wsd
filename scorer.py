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

import sys
import pandas as pd
line_answers = open("my-line-answers.txt", 'r').read().split('\n')  # sys.argv[1]
line_key = open("line-key.txt", 'r').read().split('\n')  # sys.argv[2]

# take the answers file that was generated, and reduce it to a list of the 2 senses
answers = []  # Predicted
for answer in line_answers:
    if "Phone" in answer:
        answers.append("Phone")
    if "Product" in answer:
        answers.append("Product")
# take the key file that was given, and reduce it to a list of the 2 senses
keys = []  # Actual
for key in line_key:
    if "phone" in key:
        keys.append("Phone")
    if "product" in key:
        keys.append("Product")

predicted_phone = answers.count("Phone")
predicted_product = answers.count("Product")
actual_phone = keys.count("Phone")
actual_product = keys.count("Product")

most_frequent_sense = max(actual_phone, actual_product)/len(keys)

y_actual = pd.Series(keys, name="Actual")
y_predicted = pd.Series(answers, name="Predicted")
print(pd.crosstab(y_actual, y_predicted))

# matches will return a list where each item in each list answers[] and keys[] matched.
# the length of the list will be the number of matches in the 2 lists, thus the number of times
# answers[1] and keys[1] were the same thing. (correctly predicted)

matches = [i for i, j in zip(answers, keys) if i == j]
print("Accuracy score:", (len(matches) / len(answers))*100, "%")
print(f"Most frequent sense baseline: {most_frequent_sense}%")
