import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

def ss(seed=42):
  """Function to set torch seed and cuda seed"""
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
 
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
  
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn, device):
  loss, acc = 0,0
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device), y.to(device)
      y_pred = model(X)
      loss += loss_fn(y_pred, y)
      acc += acc_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
    loss /= len(data_loader)
    acc /= len(data_loader)
    return {"model_name" : model.__class__.__name__,
            "model_loss" : loss.item(),
            "model_acc" : acc}
def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
  train_loss, train_acc = 0,0
  model.to(device)
  for batch, (X,y) in enumerate(data_loader):
    # send to GPU:
    X,y = X.to(device), y.to(device)
    # 1 forward pass
    y_pred = model(X)
    # 2 calc loss
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
    # 3 optimizer zero grad
    optimizer.zero_grad()
    # 4 loss backward
    loss.backward()
    # 5 optimizer step
    optimizer.step()
  # Calc loss and accuracy per epoch and display
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:<.5f} | Train accuracy: {train_acc:<.2f}%")

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
  test_loss, test_acc = 0,0
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device), y.to(device)
      test_pred = model(X)
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true=y, y_pred = test_pred.argmax(dim=1))
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:<.5f} | Test accuracy: {test_acc:<.2f}%")

def lazy_eval(model: torch.nn.Module, 
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, 
              optimizer: torch.optim.Optimizer,
              accuracy_fn,
              epochs: int=5,
              device: torch.device = device,
              
):
  ss()
  start_time = timer()
  for epoch in tqdm(range(epochs)):
    print(f"------->\nEpoch: {epoch}")
    train_step(data_loader=train_dataloader,
              model=model, loss_fn=loss_fn, optimizer=optimizer,
              accuracy_fn=accuracy_fn)
    test_step(data_loader=test_dataloader, model=model,
              loss_fn=loss_fn, accuracy_fn=accuracy_fn)
  displayTrainTime(start_time, timer(), device)

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample= torch.unsqueeze(sample, dim=0).to(device)
      pred_logit = model(sample)
      pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
      pred_probs.append(pred_prob.cpu())

  return torch.stack(pred_probs)
