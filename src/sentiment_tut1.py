import json as j
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer


# --- read and transform json file
json_data = None
with open('../data/yelp_academic_dataset_review.json') as data_file:
    lines = data_file.readlines()
    joined_lines = "[" + ",".join(lines) + "]"

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data)
print(data.head())


# --- prepare the data

data = data[data.stars != 3]
data['sentiment'] = data['stars'] >= 4
print(data.head())

# --- build the model

X_train, X_test, y_train, y_test = train_test_split(data, data.sentiment, test_size=0.2)

# -
count = CountVectorizer()
temp = count.fit_transform(X_train.text)

tdif = TfidfTransformer()
temp2 = tdif.fit_transform(temp)

text_regression = LogisticRegression()
model = text_regression.fit(temp2, y_train)

prediction_data = tdif.transform(count.transform(X_test.text))

predicted = model.predict(prediction_data)

# instead of doing all this steps above one could also use Pipeline
# this is a more compact way of writing the code above...
# it also has the benefit that there is no need to perform the transformations on the test data
#
#
#from sklearn.pipeline import Pipeline
#text_regression = Pipeline([('count', CountVectorizer()), ('tfidf', TfidfTransformer()),('reg', LogisticRegression())])
#
#model = text_regression.fit(X_train.text, y_train)
#predicted = model.predict(X_test.text)


# --- make predictions


print(np.mean(predicted == y_test))

# --- have some fun with the model

print(model.predict(tdif.transform(count.transform(["this product was a great video game"]))))