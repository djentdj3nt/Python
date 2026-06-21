# Фильтрация спама
# Бинарная классификация
# Векторизация

# столбцы = слова (в тексте)
# строки = образцы текста
# ячейка = кол-во данных слов в данном тексте

# очистка: строчные, удаляют знаки препинания, (стоп-слова)

import numpy as np
import pandas as pd

data = pd.read_csv("./digital_python-25-26/data/spam.csv")

print(data.columns)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data["Message"])
w = vectorizer.get_feature_names_out()

print(w)
print(w[1000])

print(X)
print(X[1000])

from sklearn.model_selection import train_test_split

X_tr, X_tst, y_tr, y_tst = train_test_split(
    data["Message"],
    data["Category"],
    test_size=0.25
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

md = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("nb", MultinomialNB())
])

md.fit(X_tr, y_tr)

texts = [
    "Hi! How are you?",
    "Win the lottery",
    "Free subsc"
]

print(md.predict(texts))

# --------------------------------------------------
# Phishing

import numpy as np
import pandas as pd

data = pd.read_csv("digital_python-25-26/data/phishing.csv")
print(data.head())

print(data.columns)

X = data.drop(columns=["class"])
print(X.head())

y = data["class"]

from sklearn.model_selection import train_test_split

X_tr, X_tst, y_tr, y_tst = train_test_split(
    X, y, test_size=0.25
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

model = dt.fit(X_tr, y_tr)

predict = model.predict(X_tst)

from sklearn.metrics import accuracy_score

print(accuracy_score(predict, y_tst))

# Классификации: бинарные(двоичные), мультиклассовые, многометочные
# - точность (precision)
# - полнота (recall)
# - specificity
# - sensitivity
# - F1-мера

# --------------------------------------------------
# Аномалии

import numpy as np
import pandas as pd

data = pd.read_csv("digital_python-25-26/data/creditcard.csv")
print(data.head())

legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]

from sklearn.model_selection import train_test_split

X_tr, X_tst, y_tr, y_tst = train_test_split(
    X, y, test_size=0.25
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ConfusionMatrixDisplay.from_estimator(
#     model1,
#     X_tst,
#     y_tst,
#     display_labels=["Легитимная", "Мошенническая"],
# )
# plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(X_tr, y_tr)

ConfusionMatrixDisplay.from_estimator(
    model1,
    X_tst,
    y_tst,
    display_labels=["Легитимная", "Мошенническая"],
)

plt.show()

from sklearn.ensemble import GradientBoostingClassifier

model3 = RandomForestClassifier()
model3.fit(X_tr, y_tr)

ConfusionMatrixDisplay.from_estimator(
    model3,
    X_tst,
    y_tst,
    display_labels=["Легитимная", "Мошенническая"],
)

plt.show()
