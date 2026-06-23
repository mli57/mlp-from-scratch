import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.layers import Linear
from src.activations import ReLU
from src.model import MLP
from src.losses import CrossEntropyLoss
from src.optimizers import SGD


# load MNIST 
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# build model to predict correct numbers (0-9)
model = MLP([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
])

loss_fn = CrossEntropyLoss()
optimizer = SGD(model.layers, lr=0.1)


# training loop
epochs = 20
batch_size = 64
loss_history = []

for epoch in range(epochs):
    # shuffle to ensure randomness
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    epoch_loss = 0
    num_batches = len(X_train) // batch_size

    for i in range(num_batches):
        xb = X_train[i*batch_size: (i+1)*batch_size]
        yb = y_train[i*batch_size: (i+1)*batch_size]

        # forward
        logits = model.forward(xb)
        loss = loss_fn.forward(logits, yb)

        # backward
        grad = loss_fn.backward()
        model.backward(grad)

        # update weights
        optimizer.step()
        epoch_loss += loss

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)

    # accuracy on test set
    logits_test = model.forward(X_test)
    preds = np.argmax(logits_test, axis = 1)
    acc = (preds == y_test).mean()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {acc*100:.1f}%")


# plot loss curve
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")
plt.show()
print("Saved loss_curve.png")