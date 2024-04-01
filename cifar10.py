import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class IntermediateBlock(nn.Module):
    def __init__(self, in_channels, num_conv_layers, conv_params):
        super(IntermediateBlock, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels, *conv_params) for _ in range(num_conv_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(conv_params[0]) for _ in range(num_conv_layers)])
        out_channels = conv_params[0]
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch_size = x.size(0)
        channel_means = x.mean(dim=[2, 3])
        a = self.fc(channel_means)
        x_out = torch.stack([F.leaky_relu(conv(x)) for conv in self.conv_layers], dim=-1).sum(dim=-1)
        x_out = torch.stack([bn(x_out) for bn in self.batch_norms], dim=-1).sum(dim=-1)
        return x_out * F.leaky_relu(a.view(batch_size, -1, 1, 1))

class OutputBlock(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_sizes=[]):
        super(OutputBlock, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(in_channels, hidden_sizes[0])] + [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)] + [nn.Linear(hidden_sizes[-1], num_classes)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes])

    def forward(self, x):
        channel_means = x.mean(dim=[2, 3])
        out = F.leaky_relu(channel_means)
        for fc, bn in zip(self.fc_layers, self.batch_norms):
            out = F.leaky_relu(bn(fc(out)))
        return out

class CustomCIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCIFAR10Net, self).__init__()
        self.intermediate_blocks = nn.ModuleList([
            IntermediateBlock(3, 3, [64, 3, 3, 1, 1]),
            IntermediateBlock(64, 3, [128, 3, 3, 1, 1]),
            IntermediateBlock(128, 3, [256, 3, 3, 1, 1]),
            IntermediateBlock(256, 3, [512, 3, 3, 1, 1]),
            IntermediateBlock(512, 3, [1024, 3, 3, 1, 1])
        ])
        self.output_block = OutputBlock(1024, num_classes, [512, 256])
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        for block in self.intermediate_blocks:
            x = block(x)
            x = self.dropout(x)
        x = self.output_block(x)
        return x

model = CustomCIFAR10Net()
input_data = torch.randn(batch_size, 3, 32, 32)
output = model(input_data)
print(output.shape)

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

print("Train Losses:" + str(train_losses[-1]))
print("Last Train Accuracy:" + str(train_accuracies[-1]))
print("Last Test Accuracy:" + str(test_accuracies[-1]))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss per Training Batch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracies')
plt.legend()

plt.tight_layout()
plt.show()

