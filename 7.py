import numpy as np
import pandas as pd

rng = np.random.default_rng(12345)

mi1 = pd.MultiIndex.from_product([["A", "B"], [2025, 2026]])
mi2 = pd.MultiIndex.from_product([["C", "D"], [2025, 2026]])
mi = pd.concat([mi1, mi2], axis=1)
print(mi)

data = rng.random((4, 6))
df = pd.DataFrame(data, index=mi, columns=mi)
print(df)

# По столбцам
print(df["B"])
print(df["B", 2025])

# Срезы
print(df.loc[:, ("B", 2025)])

# По нескольким индексам
print(df.loc[("A", 2025), ("B", 2026)])

data = {
    ("A1", 2025): 1,
    ("A2", 2026): 2,
    ("B1", 2025): 3,
    ("B2", 2026): 4,
    ("C1", 2025): 5,
    ("C2", 2026): 6,
    ("D1", 2025): 7,
    ("D2", 2026): 8,
    ("E1", 2025): 9,
    ("E2", 2026): 10,
    ("F1", 2025): 11,
    ("F2", 2026): 12,
    ("G1", 2025): 13,
    ("G2", 2026): 14,
    ("H1", 2025): 15,
    ("H2", 2026): 16,
}

sr = pd.Series(data)
print(sr)
sr.index.names = ["A", "B"]
print(sr)

print(sr["A1"])
print(sr["A1", 2025])
print(sr["A1", 2026])

# Сводные таблицы
import seaborn as sns

titanic = sns.load_dataset("titanic")
print(type(titanic))

print(titanic.head())
print(titanic.tail())

print(titanic.groupby("class").mean())

print(titanic.groupby("class").mean().reset_index())

print(titanic.groupby("class").mean().reset_index().sort_values("age", ascending=False))

births = pd.read_csv("births.csv")
print(births.head())

print(births.pivot_table("births", index="month", columns="year", aggfunc=sum))

import matplotlib.pyplot as plt

births.pivot_table("births", index="month", columns="year", aggfunc=sum).plot()
plt.show()

print(births.pivot_table("births", index="month", columns="year", aggfunc=sum))
