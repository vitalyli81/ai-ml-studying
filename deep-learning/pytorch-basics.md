# PyTorch Basics

## What Is It?

PyTorch is the **most popular framework** for building and training neural networks. It's Python-first, intuitive, and used by most researchers and companies (Meta, OpenAI, etc.).

Think of it as the **React of deep learning** — the dominant framework that most new projects use.

## Frontend Analogy — PyTorch is Like React

| React | PyTorch |
|-------|---------|
| Components | Modules (nn.Module) |
| Props | Tensors (data flowing through) |
| State | Parameters (learned weights) |
| render() | forward() |
| Virtual DOM diffing | Autograd (automatic gradients) |
| npm packages | torchvision, transformers |
| Create React App | Pre-built architectures |

## Tensors — The Core Data Structure

A tensor is a **multi-dimensional array** (like numpy arrays, but they run on GPUs).

```python
import torch

# Creating tensors (like JavaScript arrays, but with superpowers)
scalar = torch.tensor(42)                    # 0D — single number
vector = torch.tensor([1, 2, 3])             # 1D — list of numbers
matrix = torch.tensor([[1, 2], [3, 4]])      # 2D — table
image  = torch.randn(3, 224, 224)            # 3D — [channels, height, width]
batch  = torch.randn(32, 3, 224, 224)        # 4D — [batch, channels, H, W]
```

### Common Tensor Operations

```python
# Create
x = torch.zeros(3, 3)          # all zeros
x = torch.ones(3, 3)           # all ones
x = torch.randn(3, 3)          # random normal distribution
x = torch.arange(0, 10)        # [0, 1, 2, ..., 9]

# Shape operations
x = torch.randn(2, 3, 4)
x.shape                         # torch.Size([2, 3, 4])
x.reshape(6, 4)                 # reshape to [6, 4]
x.unsqueeze(0)                  # add dimension: [1, 2, 3, 4]
x.squeeze()                     # remove dimensions of size 1

# Math (element-wise, like numpy)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
a + b                           # [5, 7, 9]
a * b                           # [4, 10, 18]
a @ b                           # dot product: 32.0
torch.matmul(a, b)              # same as @

# Convert to/from numpy
numpy_array = tensor.numpy()
tensor = torch.from_numpy(numpy_array)
```

### GPU Support (The Reason PyTorch Exists)

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Move tensor to GPU
x = torch.randn(1000, 1000).to(device)

# Move model to GPU
model = model.to(device)

# In Apple Silicon Macs:
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

## nn.Module — Building Models

Every model in PyTorch is an `nn.Module`. Think of it like a React component:

```python
import torch.nn as nn

# Method 1: nn.Sequential (simple, like a pipeline)
model = nn.Sequential(
    nn.Linear(784, 256),    # input → hidden
    nn.ReLU(),
    nn.Linear(256, 128),    # hidden → hidden
    nn.ReLU(),
    nn.Linear(128, 10),     # hidden → output
)

# Method 2: Custom class (for complex architectures)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers (like constructor in React)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Define how data flows (like render in React)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = MyModel()
print(model)  # see the architecture
```

## The Complete Training Loop

This is the pattern you'll write hundreds of times:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. PREPARE DATA
X_train = torch.randn(1000, 10)       # 1000 samples, 10 features
y_train = torch.randint(0, 3, (1000,)) # 3 classes

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. DEFINE MODEL
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 3),
)

# 3. DEFINE LOSS + OPTIMIZER
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 4. TRAINING LOOP
model.train()  # enable dropout, batch norm in training mode

for epoch in range(20):
    total_loss = 0

    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

# 5. EVALUATION
model.eval()  # disable dropout, batch norm in eval mode

with torch.no_grad():  # no gradient computation needed for inference
    test_predictions = model(X_test)
    predicted_classes = test_predictions.argmax(dim=1)
```

## DataLoader — Batching Made Easy

```python
from torch.utils.data import DataLoader, Dataset

# Custom dataset (like a custom data hook)
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader handles batching, shuffling, parallel loading
loader = DataLoader(
    MyDataset(X_train, y_train),
    batch_size=32,           # process 32 samples at a time
    shuffle=True,            # randomize order each epoch
    num_workers=4,           # parallel data loading
)

# Iterate over batches
for batch_X, batch_y in loader:
    print(batch_X.shape)     # [32, num_features]
    print(batch_y.shape)     # [32]
```

## Saving and Loading Models

```python
# Save the model weights
torch.save(model.state_dict(), 'model.pt')

# Load into a new model instance
model = MyModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()  # set to evaluation mode

# Save everything (model + optimizer + epoch) for resuming training
checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')
```

## model.train() vs model.eval()

This trips up beginners. You **must** switch modes:

```python
model.train()  # TRAINING mode
# - Dropout is active (randomly zeros neurons)
# - BatchNorm uses batch statistics
# - Gradients are computed

model.eval()   # EVALUATION mode
# - Dropout is disabled (all neurons active)
# - BatchNorm uses running statistics
# - Usually wrapped with torch.no_grad()

# Typical pattern:
model.train()
for epoch in range(num_epochs):
    # ... training loop ...

model.eval()
with torch.no_grad():
    # ... validation / prediction ...
```

## Common Layer Types Cheat Sheet

```python
# Dense / Fully Connected
nn.Linear(in_features, out_features)

# Convolution (for images)
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
nn.MaxPool2d(kernel_size=2)

# Recurrent (for sequences)
nn.LSTM(input_size, hidden_size, num_layers=2)
nn.GRU(input_size, hidden_size)

# Normalization
nn.BatchNorm1d(num_features)     # for dense layers
nn.BatchNorm2d(num_channels)     # for conv layers
nn.LayerNorm(normalized_shape)   # for transformers

# Regularization
nn.Dropout(p=0.3)

# Activation
nn.ReLU()
nn.GELU()
nn.Sigmoid()

# Transformer
nn.TransformerEncoder(encoder_layer, num_layers)
nn.MultiheadAttention(embed_dim, num_heads)

# Embedding (words → vectors)
nn.Embedding(vocab_size, embed_dim)
```

## Debugging Tips

```python
# Check tensor shapes (most common bug: shape mismatch)
print(f"Input shape: {x.shape}")
print(f"After layer1: {self.layer1(x).shape}")

# Check for NaN (training exploded)
print(torch.isnan(loss).any())

# Check gradients exist
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean():.6f}")

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,}")
```

## PyTorch vs TensorFlow

| PyTorch | TensorFlow |
|---------|------------|
| Python-first, intuitive | More boilerplate |
| Dynamic computation graph | Static (eager mode available) |
| Debugging with regular Python | Harder to debug |
| Research standard | Production/deployment standard |
| Meta, OpenAI, Anthropic | Google |
| **Learn this first** | Learn later if needed |

## Key Takeaway

PyTorch is **4 things**: tensors (data), nn.Module (model building), autograd (automatic gradients), and DataLoader (batching). The training loop is always the same: forward pass → compute loss → backward pass → update weights. Learn this pattern by heart — you'll use it in every project. Start with `nn.Sequential` for simple models, move to custom `nn.Module` classes when you need more control.
