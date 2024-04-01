import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import batch_size
from dataloader import data_transform, data_loader
from blocknet10 import CustomCIFAR10Net
from analytics import model_analytics

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_train, transform_test = data_transform()
train_loader, test_loader = data_loader(transform_train, transform_test)

def arch_tester():
    model = CustomCIFAR10Net()
    input_data = torch.randn(batch_size, 3, 32, 32)
    output = model(input_data)
    return output.shape

arch_tester_output = arch_tester()
print(arch_tester_output)

model = CustomCIFAR10Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.8, 0.95), weight_decay=0.0005, amsgrad=True, eps=1e-8)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_losses = []
train_accuracies = []
test_accuracies = []

num_epochs = 50
total_steps = len(train_loader) * num_epochs
step_count = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step_count += 1
        if step_count % 100 == 0:
            train_losses.append(running_loss / 100)
            print(f'[Epoch: {epoch + 1}, Step: {step_count:5d}/{total_steps}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

        if i == len(train_loader) - 1:
            model.eval()
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)

    scheduler.step()

    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch + 1}: Test Accuracy = {test_accuracy:.2f}%')


print("Last Train Losses:" + str(train_losses[-1]))
print("Last Train Accuracy:" + str(train_accuracies[-1]))
print("Last Test Accuracy:" + str(test_accuracies[-1]))

analytics = model_analytics(train_losses, train_accuracies, test_accuracies)