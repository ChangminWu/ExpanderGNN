import torch

from train.metrics import MAE
from utils import check_tensorboard


def train_epoch_sparse(model, optimizer, device, data_loader, epoch,
                       writer=None):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()

        try:
            batch_pos_enc = batch_graphs.ndata["pos_enc"].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x,
                                         batch_e, batch_pos_enc)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)

        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)

    if writer is not None:
        writer, _ = check_tensorboard(model, writer, step=epoch)

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    return epoch_loss, epoch_train_mae, optimizer, writer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x,
                                             batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)

    return epoch_test_loss, epoch_test_mae


def check_patience(all_losses, best_loss, best_epoch,
                   curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter
