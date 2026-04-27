import numpy as np
import pandas as pd

print(pd.date_range("2026-01-01", periods=4, freq="ME"))
print(pd.date_range("2026-01-01", periods=4, freq="MS"))
print(pd.date_range("2026-01-01", periods=4, freq="QE"))
print(pd.date_range("2026-01-01", periods=4, freq="QS"))
print(pd.date_range("2026-01-01", periods=4, freq="W"))

ind = pd.read_csv(
    "./digital_python-25-26/data/index.csv",
    sep=";",
    parse_dates=["Date"]
)

print(ind.head())
print(type(ind))
print(ind.dtypes)

index = pd.DatetimeIndex(ind["Date"])
ind.index = index

print(ind.head())

ind = ind["Close"]
print(ind.head())

import matplotlib.pyplot as plt

ind.plot()
plt.show()

# Данные по велосипедистам

df = pd.read_csv(
    "./digital_python-25-26/data/FremontBridge.csv",
    index_col="Date",
    parse_dates=True,
    date_format="%Y-%m-%d %I:%M:%S %p"
)

print(df.head())
print(df.dtypes)

print(df.describe())
print(df.dropna().describe())

# df.plot()
# plt.ylabel("Кол-во велосипедистов (в час)")
# plt.show()

weekly = df.resample("W").sum()
weekly.plot(style=["-", ":", "--"])
plt.ylabel("Кол-во велосипедистов (по неделям)")
plt.show()

# daily = df.resample("D").sum()
# daily.rolling(30, center=True, win_type="gaussian").mean(std=5).plot(
#     style=["-", ":", "--"]
# )
# plt.ylabel("Среднее месячное кол-во велосипедистов (скользящее окно)")
# plt.show()

# Среднее значение по времени суток

timely = df.groupby(df.index.time).mean()
ticks = 60 * 60 * 4 * np.arange(6)
print(ticks)
timely.plot(xticks=ticks)
plt.show()

# Среднее значение по дням недели

weekly = df.groupby(df.index.dayofweek).mean()
weekly.plot()
plt.show()

# Разделение на будни и выходные

w1 = np.where(df.index.weekday < 5, "Будни", "Выходные")
t1 = df.groupby([w1, df.index.time]).mean()

fig, ax = plt.subplots(1, 2)

ax[0].set_ylim(0, 600)
t1.loc["Будни"].plot(ax=ax[0], title="Будни")

ax[1].set_ylim(0, 600)
t1.loc["Выходные"].plot(ax=ax[1], title="Выходные")

plt.show()

df.plot()
plt.ylabel("Кол-во велосипедистов (в час)")
plt.show()
