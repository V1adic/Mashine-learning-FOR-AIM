import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as Function
from torch import optim
import matplotlib.pyplot as plt

device = torch.device("cpu")

def get_df(path = 'Data/data.csv'):
    df = pd.read_csv(path).dropna()
    # Преобразует столбцы RainToday и RainTomorrow в целые числа
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1}).astype(int)
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1}).astype(int)
    return df

# Конвертирует данные в тензоры PyTorch
def convert_X_to_tensor(X, is_scaled):
    return torch.from_numpy(X.to_numpy() if not is_scaled else X).float()

def convert_y_to_tensor(y, is_scaled):
    return torch.squeeze(torch.from_numpy(y.to_numpy() if not is_scaled else y).float())

# Создаёт массивы X и y из выбранных столбцов данных
def get_X_y_tensor(df, is_scaled = False):
    X = df[['Rainfall', 'Humidity3pm', 'RainToday', 'Pressure9am']]
    y = df[['RainTomorrow']]

    if is_scaled:
        scaler = MinMaxScaler() # При необходимости масштабирует данные
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Делит данные на тренировочный и тестовый наборы

   # Перемещает тензоры на устройство
    X_train = convert_X_to_tensor(X_train, is_scaled).to(device)
    y_train = convert_y_to_tensor(y_train, is_scaled).to(device)
    X_test = convert_X_to_tensor(X_test, is_scaled).to(device)
    y_test = convert_y_to_tensor(y_test, is_scaled).to(device)

    return X_train, X_test, y_train, y_test

def calculate_accuracy(y_true, y_pred):

    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

def calc_by_epochs(X_train, X_test, y_train, y_test, num_epochs, net, optimizer, criterion = nn.BCELoss().to(device)): # Точность по эпохам
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        y_pred = net(X_train)
        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, y_train) 

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:

            y_test_pred = net(X_test)
            y_test_pred = torch.squeeze(y_test_pred)

            test_loss = criterion(y_test_pred, y_test) # Вычисляем тестовую потерю
            test_acc = calculate_accuracy(y_test, y_test_pred) 

            print(f'Эпоха {epoch}\nТестовый набор - потеря: {test_loss}, точность: {test_acc}')
            epoch_losses.append(test_loss)
            epoch_accuracies.append(test_acc)

    return epoch_losses, epoch_accuracies

def calc_by_learning_rates(X_train, X_test, y_train, y_test, learning_rates, num_epochs):
    results = {}

    for lr in learning_rates:
        print(f'learning_rate={lr}')

        net = Net(4).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr) 

        epoch_losses, epoch_accuracies = calc_by_epochs(X_train, X_test, y_train, y_test, num_epochs, net, optimizer)

        results[lr] = {
            'потери': epoch_losses,
            'точности': epoch_accuracies,
        }

    return results

def calc_by_architectures(X_train, X_test, y_train, y_test, architectures, optim_learning_rate, num_epochs):
    results = {}

    for architecture in architectures:
        print(f'Тестовая архитектура: {architecture}')

        net = Net(4, architecture).to(device) 
        optimizer = optim.Adam(net.parameters(), lr=optim_learning_rate) 

        epoch_losses, epoch_accuracies = calc_by_epochs(X_train, X_test, y_train, y_test, num_epochs, net, optimizer)

        results[str(architecture)] = {
            'потери': epoch_losses,
            'точности': epoch_accuracies,
        }

    return results

def calc_by_activation_functions(X_train, X_test, y_train, y_test, activation_functions, optim_learning_rate, num_epochs):
    results = {}

    for activation_fn in activation_functions:
        print(f'Функция активации: {activation_fn.__name__}')

        net = Net(4, activation_fn).to(device) 
        optimizer = optim.Adam(net.parameters(), lr=optim_learning_rate) 

        # Запуск функции calc_by_epochs для обучения модели на num_epochs эпох.
        epoch_losses, epoch_accuracies = calc_by_epochs(X_train, X_test, y_train, y_test, num_epochs, net, optimizer)

        results[activation_fn.__name__] = {
            'потери': epoch_losses,
            'точности': epoch_accuracies,
        }

    return results

def calc_by_optimizers(X_train, X_test, y_train, y_test, optimizers, optim_learning_rate, optim_fn, num_epochs):
    results = {}

    for optimizer_name, optimizer_class in optimizers:
        print(f'Оптимайзер: {optimizer_name}')

        net = Net(4, optim_fn).to(device)
        optimizer = optimizer_class(net.parameters(), lr=optim_learning_rate)

        epoch_losses, epoch_accuracies = calc_by_epochs(X_train, X_test, y_train, y_test, num_epochs, net, optimizer)

        results[optimizer_name] = {
            'потери': epoch_losses,
            'точности': epoch_accuracies,
        }

    return results

def initialize_weights(model, init_type):

    match init_type:
        case "random":
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
        case 'xavier':
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
        case 'he':
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)
        case 'zero':
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    return model

def calc_by_weight_initialization(X_train, X_test, y_train, y_test, init_types, optim_learning_rate, optim_fn, num_epochs):
    results = {}

    for init_type in init_types:
        print(f'Инициализация весов: {init_type}')

        net = Net(4, optim_fn).to(device)
        net = initialize_weights(net, init_type)
        optimizer = optim.Adam(net.parameters(), lr=optim_learning_rate)

        epoch_losses, epoch_accuracies = calc_by_epochs(X_train, X_test, y_train, y_test, num_epochs, net, optimizer)

        results[init_type] = {
            'потери': epoch_losses,
            'точности': epoch_accuracies,
        }

    return results

def calc_loss(X_train, X_test, y_train, y_test,
              num_epochs = 2000, learning_rates = None, architectures = None,
              activation_functions = None, optimizers = None, init_types = None,
              optim_lr = None, optim_fn = None):

    if learning_rates != None: 
        results = calc_by_learning_rates(X_train, X_test, y_train, y_test, learning_rates, num_epochs)

    elif architectures != None: 
        results = calc_by_architectures(X_train, X_test, y_train, y_test, architectures, optim_lr, num_epochs)

    elif activation_functions != None:
        results = calc_by_activation_functions(X_train, X_test, y_train, y_test, activation_functions, optim_lr, num_epochs)

    elif optimizers != None: 
        results = calc_by_optimizers(X_train, X_test, y_train, y_test, optimizers, optim_lr, optim_fn, num_epochs)

    else: 
        results = calc_by_weight_initialization(X_train, X_test, y_train, y_test, init_types, optim_lr, optim_fn, num_epochs)

    return results

def visualize_results(results): # Ploting
    plt.figure(figsize=(6, 4))

    for param, metrics in results.items():
        losses = [loss.detach().cpu().numpy() for loss in metrics["потери"]]
        accuracies = [acc.detach().cpu().numpy() for acc in metrics["точности"]]

        epochs = range(0, len(losses) * 100, 100)

        fig, ax1 = plt.subplots(figsize=(6, 4))

        ax1.set_xlabel('Эпохи', fontsize=14)
        ax1.set_ylabel('Потеря', fontsize=14, color='tab:red')
        ax1.plot(epochs, losses, label=f'Потеря ({param})', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_yscale('log')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Точность', fontsize=14, color='tab:blue')
        ax2.plot(epochs, accuracies, label=f'Точность ({param})', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        fig.suptitle(f"Конвергенция для {param}", fontsize=16)
        fig.tight_layout()
        plt.legend(loc='upper left', fontsize=12)
        plt.show()

class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__() # Вызов родительского конструктора класса
    self.fc1 = nn.Linear(n_features, 8) # 3 линейных слоя 
    self.fc2 = nn.Linear(8, 4) 
    self.fc3 = nn.Linear(4, 1)

  def forward(self, x): # Прохождение слоёв
    x = Function.relu(self.fc1(x)) 
    x = Function.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))


# Main
df = get_df()
X_train, X_test, y_train, y_test = get_X_y_tensor(df)
learning_rates = [0.0001, 0.001, 0.01, 0.1]
results = calc_loss(X_train, X_test, y_train, y_test, learning_rates=learning_rates)
visualize_results(results)

# For optim_learning_rate = 0.001
optim_learning_rate = 0.001
X_train, X_test, y_train, y_test = get_X_y_tensor(df, is_scaled=True)
results = calc_loss(X_train, X_test, y_train, y_test, learning_rates=learning_rates)
visualize_results(results)


# For optim_learning_rate = 0.1
optim_learning_rate = 0.1

class Net(nn.Module): # Owerride Net class.
    def __init__(self, n_features, architecture):
        super(Net, self).__init__()

        # Создание слоев на основе переданной архитектуры
        layers = []
        input_size = n_features
        for layer_size in architecture:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            input_size = layer_size
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
architectures = [
    [8, 4],
    [16, 8, 4],
    [32, 16, 8, 4],
    [4],
    [8],
    [],
]

results = calc_loss(X_train, X_test, y_train, y_test, architectures=architectures, optim_lr=optim_learning_rate)
visualize_results(results)

class Net(nn.Module): # Owerride Net class.
    def __init__(self, n_features, activation_fn):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.activation_fn = activation_fn # Отличие от первоначального (да, можно оптимизировать)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
    
activation_functions = [Function.relu, torch.sigmoid, torch.tanh, Function.leaky_relu]
results = calc_loss(X_train, X_test, y_train, y_test, activation_functions=activation_functions, optim_lr=optim_learning_rate)
visualize_results(results)
optim_activation_function = torch.sigmoid


optimizers = [
    ('SGD', optim.SGD),
    ('Adam', optim.Adam),
    ('RMSprop', optim.RMSprop),
    ('Adagrad', optim.Adagrad)
]

results = calc_loss(X_train, X_test, y_train, y_test, optimizers=optimizers, optim_lr=optim_learning_rate, optim_fn=optim_activation_function)
visualize_results(results)


init_types = ['random', 'xavier', 'he', 'zero']
results = calc_loss(X_train, X_test, y_train, y_test, init_types=init_types, optim_lr=optim_learning_rate, optim_fn=optim_activation_function)
visualize_results(results)