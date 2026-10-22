import numpy as np
import torch
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
        return self.output(x)

def generate_dataset(num_samples=1000):
    genders = np.random.choice([0, 1], size=num_samples)
    
    # Генерация роста и веса на основе пола без лишних циклов
    heights = np.where(genders == 1, 
                       np.random.normal(165, 6, num_samples), 
                       np.random.normal(178, 7, num_samples))
    
    weights = np.where(genders == 1, 
                       np.random.normal(60, 8, num_samples), 
                       np.random.normal(78, 10, num_samples))
    
    X = np.column_stack((heights.round(1), weights.round(1)))
    return X, genders

X, y = generate_dataset(1000)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  

network = NeuralNetwork()
def_loss = nn.BCEWithLogitsLoss()
def_optimize = optim.Adam(network.parameters(), lr=0.01)

loss_history = []

for epoch in range(500):
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
    print(f"Probability of male: {1 - prob:.3f}")
    print("Predicted gender:", "Female" if gender == 1 else "Male")

    again = input("\nDo you want to test another person? (y/n): ")
    if again.lower() != "y":
        print("Exit.")
        break