import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Генерация набора данных временных последовательностей
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Создаем синусоидальный набор данных
seq_length = 20
n_samples = 1000
x = np.linspace(0, 100, n_samples)
data = np.sin(x) + 0.1 * np.random.normal(size=len(x))  # Добавляем шум для усложнения задачи

# Разделение на тренировочный и тестовый наборы данных
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Определение модели RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # Инициализация скрытого состояния
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Используем только последний выход
        return out
    
# Модифицированная модель с LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # Инициализация скрытого состояния
        c0 = torch.zeros(1, x.size(0), self.hidden_size)  # Инициализация состояния ячейки
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Используем только последний выход
        return out

input_size = 1
hidden_size = 50
output_size = 1

def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.view(-1, seq_length, input_size)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Обучение RNN...")
rnn_model = RNN(input_size, hidden_size, output_size)
train_model(rnn_model, train_dataloader, epochs=20)

print("\nОбучение LSTM...")
lstm_model = LSTM(input_size, hidden_size, output_size)
train_model(lstm_model, train_dataloader, epochs=20)

# Проверка модели на тестовом наборе данных
model = lstm_model
model.eval()
predictions = []
actual = []

with torch.no_grad():
    for x, y in test_dataloader:
        x = x.view(1, seq_length, input_size)
        pred = model(x).item()
        predictions.append(pred)
        actual.append(y.item())

plt.figure(figsize=(10, 6))
plt.plot(actual, label="Actual", linestyle='-', color='black')
plt.plot(predictions, label="Predicted", linestyle='--', color='blue')
plt.legend()
plt.xlabel("Временные шаги")
plt.ylabel("Значение")
plt.title("Фактическое значение по сравнению с Прогнозируемым по тестовым данным")
plt.show()

# Разделить данные на N групп. (возможно, выбрать первые M данных для обучения)
N = 2

df = pd.read_csv('Data/ETTm1.csv')
dfs = np.array_split(df, N)

# Реализовать предсказание ARIMA для 2 (тренировочной группы) на основе 1. Рассчитать MSE
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

train = dfs[0]["OT"]
test = dfs[1]["OT"]

model = ARIMA(train, order=(3,0,0))
model_fit = model.fit()

predictions = model_fit.forecast(steps=len(test))

mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Исследовать качество работы модели в зависимости от параметров (для групп 1-2).

# Параметры для исследования (p, d, q)
p_values = range(0, 4)
d_values = range(0, 2)
q_values = range(0, 4)

results = []

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()

                predictions = model_fit.forecast(steps=len(test))

                mse = mean_squared_error(test, predictions)

                print(f'Parameters (p={p}, d={d}, q={q}): mse = {mse}')
                results.append((p, d, q, mse))

            except Exception as e:
                print(f"Error with parameters (p={p}, d={d}, q={q}): {e}")

results_df = pd.DataFrame(results, columns=['p', 'd', 'q', 'MSE'])
best_params = results_df.loc[results_df['MSE'].idxmin()]

print("Best Parameters (p, d, q):\n", best_params[['p', 'd', 'q']])
print("Best MSE:", best_params['MSE'])

results_df = results_df.groupby(["p", "d", "q"], as_index=False)["MSE"].mean()

d_values = results_df["d"].unique()

# Настройка 3D графика
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for d in d_values:
    subset_df = results_df[results_df["d"] == d]
    p_values = subset_df["p"]
    q_values = subset_df["q"]
    mse_values = subset_df["MSE"]

    ax.scatter(p_values, q_values, mse_values, label=f'd={d}', alpha=0.6)

ax.set_xlabel('p')
ax.set_ylabel('q')
ax.set_zlabel('MSE')
ax.legend(title="d")
ax.set_title("3D Visualization of MSE for ARIMA parameters (p, q, d)")
plt.show()

# Выполнить итеративное предсказание, данные для каждого следующего предсказания обновлять по методу "экспоненциальное среднее".

alpha = 0.5  # Коэффициент сглаживания для экспоненциального среднего

model = ARIMA(train, order=(1, 0, 0)).fit()

predictions = []
current_prediction = model.forecast(steps=1).iloc[0]  # Первое предсказание

for true_value in test:
    predictions.append(current_prediction)
    current_prediction = alpha * true_value + (1 - alpha) * current_prediction

mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error (Iterative with Exponential Smoothing): {mse}')