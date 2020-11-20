import torch

from utils import check_tensorboard

cls_criterion = torch.nn.BCEWithLogitsLoss()


def train_epoch(model, optimizer, device, evaluator, data_loader, epoch, writer=None):
    model.train()
    epoch_loss = 0
    y_true = []
    y_pred = []
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        batch_scores = model.forward(batch_graphs, batch_x.to(torch.float32), batch_e.to(torch.float32))
        is_labeled = batch_labels == batch_labels
        loss = cls_criterion(batch_scores.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        y_true.append(batch_labels.view(batch_scores.shape).detach().cpu())
        y_pred.append(batch_scores.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if writer is not None:
        writer, _ = check_tensorboard(model, writer, step=epoch)

    epoch_loss /= (iter + 1)
    return epoch_loss, evaluator.eval(input_dict)['rocauc'], optimizer, writer


def evaluate_network(model, device, evaluator, data_loader):
    model.eval()
    epoch_test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_graphs, batch_x.to(torch.float32), batch_e.to(torch.float32))

            is_labeled = batch_labels == batch_labels
            loss = cls_criterion(batch_scores.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
            epoch_test_loss += loss.detach().item()

            y_true.append(batch_labels.view(batch_scores.shape).detach().cpu())
            y_pred.append(batch_scores.detach().cpu())

        epoch_test_loss /= (iter + 1)
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}

    return epoch_test_loss, evaluator.eval(input_dict)['rocauc']


def check_patience(best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter