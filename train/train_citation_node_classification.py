import torch

from train.metrics import accuracy_node as accuracy
from utils import check_tensorboard


def train_epoch(model, optimizer, device, graph, epoch,
                nfeat, efeat, train_mask, labels, writer=None):
    model.train()

    logits = model(graph, nfeat, efeat)
    loss = model.loss(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()
    epoch_train_acc = accuracy(logits[train_mask], labels[train_mask])

    if writer is not None:
        writer, _ = check_tensorboard(model, writer, step=epoch)
    return epoch_loss, epoch_train_acc, optimizer, writer


def evaluate_network(model, device, graph, nfeat, efeat, mask, labels):
    model.eval()
    with torch.no_grad():
        logits = model.forward(graph, nfeat, efeat)
        loss = model.loss(logits[mask], labels[mask])
        epoch_test_loss = loss.detach().item()
        epoch_test_acc = accuracy(logits[mask], labels[mask])

    return epoch_test_loss, epoch_test_acc


def check_patience(all_losses, best_loss, best_epoch,
                   curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter
