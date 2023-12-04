#import packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

#import file
data=pd.read_json("sr.json", lines=True)
print(data.head())

data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not sarcasm", 1:"sarcasm"})
print(data.head(10))

data=data[["headline", "is_sarcastic"]]
x=np.array(data["headline"])
y=np.array(data["is_sarcastic"])

cv=CountVectorizer()
x=cv.fit_transform(x)
x_train, x_test, y_train, y_test

model=BernoulliNB()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

user=input("Enter a text: ")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)