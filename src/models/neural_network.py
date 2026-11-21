import copy

import torch
import torch.nn as nn

from seeds import SEED


class MLP(nn.Module):
    """A simple multi-layer perceptron.
    Feed-forward network: n_inputs -> hidden layers (Linear + ReLU + Dropout) -> n_classes scores.

    Args:
        n_inputs (int): The number of input features.
        n_classes (int): The number of output classes.
        hidden_sizes (list, optional): Number of neurons in each hidden layer. Defaults to [256].
        dropout (float, optional): The dropout rate. Defaults to 0.0.
    """

    def __init__(self, n_inputs, n_classes, hidden_sizes=None, dropout=0.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256]
        modules = []
        current_size = n_inputs
        for n_neurons in hidden_sizes:
            modules += [nn.Linear(current_size, n_neurons), nn.ReLU(), nn.Dropout(dropout)]
            current_size = n_neurons
        modules.append(nn.Linear(current_size, n_classes))
        self.network = nn.Sequential(*modules)

    def forward(self, features):
        return self.network(features)


def train_one_epoch(model, loader, loss_fn, optimizer):
    """Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): The data loader providing training data.
        loss_fn (callable): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.

    Returns:
        None, but updates the model parameters in place.
    """
    model.train()
    for batch_features, batch_labels in loader:
        scores = model(batch_features)          # 1. Predict the scores for the batch
        loss = loss_fn(scores, batch_labels)    # 2. Compute the loss (measure the error)
        optimizer.zero_grad()
        loss.backward()                         # 3. Backpropagate the loss (compute gradients)
        optimizer.step()                        # 4. Update the model parameters (move every weight a little)


def evaluate(model, features, labels, loss_fn):
    """Evaluate the model on the given features and labels.

    Args:
        model (nn.Module): The neural network model to evaluate.
        features (torch.Tensor): The input features for evaluation.
        labels (torch.Tensor): The true labels corresponding to the features.
        loss_fn (callable): The loss function to use for evaluation.

    Returns:
        tuple: A tuple containing the loss and accuracy of the model on the given data.
    """
    model.eval()
    with torch.no_grad():
        scores = model(features)
        loss = loss_fn(scores, labels).item()
        accuracy = (scores.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def predict(model, features):
    """Predict the class labels for the given features.

    Args:
        model (nn.Module): The trained neural network model.
        features (torch.Tensor): The input features for which to predict class labels.

    Returns:
        numpy.ndarray: The predicted class labels as a NumPy array.
    """
    model.eval()
    with torch.no_grad():
        return model(features).argmax(dim=1).numpy()


def fit(train_features, train_labels, validation_features, validation_labels,
        hidden_sizes=None, dropout=0.0, lr=1e-3, weight_decay=0.0, batch_size=128, epochs=30):
    """
    Train a multi-layer perceptron (MLP) model on the provided training data and evaluate it on the validation data.

    Args:
        train_features (torch.Tensor): The input features for training.
        train_labels (torch.Tensor): The true labels for training.
        validation_features (torch.Tensor): The input features for validation.
        validation_labels (torch.Tensor): The true labels for validation.
        hidden_sizes (list): A list of integers representing the number of neurons in each hidden layer.
        dropout (float): The dropout rate to use during training.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        batch_size (int): The batch size to use during training.
        epochs (int): The number of epochs to train the model.

    Returns:
        tuple (torch.nn.Module, dict): A tuple containing the trained model and a history dictionary with training and validation loss and accuracy (model, history).
    """
    torch.manual_seed(SEED)
    n_classes = int(train_labels.max() + 1)
    model = MLP(train_features.shape[1], n_classes, hidden_sizes, dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "train_acc": [], "validation_loss": [], "validation_acc": []}
    best_weights = None
    for _ in range(epochs):
        train_one_epoch(model, loader, loss_fn, optimizer)
        train_loss, train_acc = evaluate(model, train_features, train_labels, loss_fn)
        validation_loss, validation_acc = evaluate(model, validation_features, validation_labels, loss_fn)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["validation_loss"].append(validation_loss)
        history["validation_acc"].append(validation_acc)

        if validation_acc == max(history["validation_acc"]):
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history


def train_final(features, labels, epochs, hidden_sizes=None, dropout=0.0, lr=1e-3, weight_decay=0.0, batch_size=128):
    """
    Train a multi-layer perceptron (MLP) model on the provided training data for a specified number of epochs
    (the best epoch found by fit on the validation set).

    Args:
        features (torch.Tensor): The input features for training.
        labels (torch.Tensor): The true labels for training.
        epochs (int): The number of epochs to train the model.
        hidden_sizes (list): A list of integers representing the number of neurons in each hidden layer.
        dropout (float): The dropout rate to use during training.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        batch_size (int): The batch size to use during training.

    Returns:
        model (torch.nn.Module): The trained model after the final training on the entire dataset.
    """
    torch.manual_seed(SEED)
    n_classes = int(labels.max() + 1)
    model = MLP(features.shape[1], n_classes, hidden_sizes, dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        train_one_epoch(model, loader, loss_fn, optimizer)

    return model
