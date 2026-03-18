import numpy as np
import pandas as pd

# Строковые операции

data = ["one", "twO", "three", "foUr", "fIvE"]
print(data)

print(data[0].upper())
print(data[1].lower())

print(data[2].capitalize())

print(data[3].swapcase())

print(data[4].title())

print(data[0].isupper())
print(data[1].islower())

rgex = r"([a-z0-9]+\s).*"

a = names.str.extract(rgex, expand=False)
print(a)

# Индикаторные переменные

data = ["one1 one2 one3", "two9999 ddd ", "tHREE FoUR"]
names = pd.DataFrame({"name": data, "info": ["A|B", "B|C|D", "E|F|G"]})
print(names)

print(names["info"].str.get_dummies("|"))

recipes = pd.read_json("./digital_python-25-26/data/recipeitem.json")
print(recipes.head())

print(recipes.shape)

print(recipes.iloc[0])

# Работа с временными рядами

# Python
from datetime import datetime

import dateutil

d = datetime(year=2019, month=1, day=1)
print(d)

d = dateutil.parser.parse("2019-01-01")
print(d)

# Numpy

d = np.array("2026-03-04", dtype=np.datetime64)
print(d)

# Pandas

d = pd.to_datetime("2026-03-04")
print(d)

# Коды периодичности

p1 = pd.Period("2026-03-04", freq="D")
print(p1)
