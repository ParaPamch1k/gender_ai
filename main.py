import torch, random, pandas
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
   
female_names = [
    "Alice","Natasha","Kate","Oksana","Julia","Mila","Irina","Diana","Sofia","Maria",
    "Vera","Nastya","Polina","Daria","Anna","Lydia","Elena","Zhanna","Olga","Tatyana"
]

male_names = [
    "Andrey","Sergey","Pavel","Egor","Artem","Gleb","Nikita","Oleg","Fedor","Grisha",
    "Roman","Ivan","Dmitry","Mikhail","Alexey","Vladimir","Yan","Stas","Kirill","Petr"
]

def generate_person():
    # 0 = male, 1 = female
    gender = np.random.choice([0, 1])

    if gender == 1:
        name = random.choice(female_names)
        height = round(np.random.normal(165, 6), 1)  
        weight = round(np.random.normal(60, 8), 1)   
    else:
        name = random.choice(male_names)
        height = round(np.random.normal(178, 7), 1)  
        weight = round(np.random.normal(78, 10), 1)  

    return name, height, weight, gender

dataset = [generate_person() for _ in range(1000)]

data_table = pandas.DataFrame(dataset, columns=["Name", "Height", "Weight", "Gender"])
X = data_table[["Height", "Weight"]].values
y = data_table["Gender"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  

network = NeuralNetwork()

def_loss = nn.BCEWithLogitsLoss()
def_optimize = optim.Adam(network.parameters(), lr=0.01)

loss_history = []

epochs = 500
for epoch in range(epochs):
    preds = network(X_tensor)
    loss = def_loss(preds, y_tensor)
    
    def_optimize.zero_grad()
    loss.backward()
    def_optimize.step()

    loss_history.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history, linewidth=2)
plt.title("Neural Network Training Graph (Loss)")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.grid(True)
plt.show()

def predict_gender(height, weight):
    with torch.no_grad():
        x = torch.tensor([[height, weight]], dtype=torch.float32)
        logits = network(x)
        prob = torch.sigmoid(logits).item()
        gender = 1 if prob > 0.5 else 0
        return prob, gender

while True:
    print("\nEnter data to predict gender:")
    try:
        height = float(input("Height (cm): "))
        weight = float(input("Weight (kg): "))
    except ValueError:
        print("Error: please enter numbers!")
        continue

    prob, gender = predict_gender(height, weight)

    print("\nRESULT:")
    print(f"Probability of female: {prob:.3f}")
    print("Probability of male: " + str(1 - prob))
    print("Predicted gender:", "Female" if gender == 1 else "Male")

    again = input("\nDo you want to test another person? (y/n): ")
    if again.lower() != "y":
        print("Exit.")
        break
