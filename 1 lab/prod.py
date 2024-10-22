import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/titanic.csv")

print(df)

df = df.dropna()
df_male = df[df['Sex'] == 'male']
df_female = df[df['Sex'] == 'female']

df_male_age = df_male['Age']
df_female_age = df_female['Age']

plt.hist(df_female_age, edgecolor='black')
plt.title('Распределение женщин по возрасту')
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')
plt.show()

plt.hist(df_male_age, edgecolor="blue")
plt.title('Распределение мужчин по возрасту')
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')
plt.show()

ax1 = df_male_age.plot(color='blue', label='male')
ax2 = df_female_age.plot(color='red', secondary_y=True, label='female')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.legend(h1+h2, l1+l2, loc=1)
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

print(f"Max Age from male -> {df_male_age.max()}")
print(f"Min Age from male -> {df_male_age.min()}")
print(f"Mean Age from male -> {df_male_age.mean()}")
print(f"Mode Age from male -> {df_male_age.mode().max()}")
print(f"Median Age from male -> {df_male_age.median()}", end="\n\n\n\n\n")

print(f"Max Age from female -> {df_female_age.max()}")
print(f"Min Age from female -> {df_female_age.min()}")
print(f"Mean Age from female -> {df_female_age.mean()}")
print(f"Mode Age from female -> {df_female_age.mode().max()}")
print(f"Median Age from female -> {df_female_age.median()}", end="\n\n\n\n\n")