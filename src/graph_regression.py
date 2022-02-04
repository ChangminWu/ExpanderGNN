import argparse
import os
import os.path as osp
import logging

import torch
import torch.nn.functional as F

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from logger import Logger
from models import ExpanderPNA, PNA
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime
from time import time

from utils import Evaluator


def train(model, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Graph-Regression')
    parser.add_argument('--dataset', type=str, default="ZINC")
    parser.add_argument('--outdir', type=str, default="./output/")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-steps', type=int, default=1)

    parser.add_argument('--use-expander', action='store_true')
    parser.add_argument('--use-activation', action='store_true')
    parser.add_argument('--num-readout-layers', type=int, default=1)
    parser.add_argument('--hiddim', type=int, default=256)

    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--sample-method', type=str, default='prabhu')
    parser.add_argument('--weight-initializer', type=str, default='glorot')
    parser.add_argument('--dense-output', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    if args.weight_initializer == 'default':
        args.weight_initializer = None

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
    train_dataset = ZINC(path, subset=True, split='train')
    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_expander:
        model = ExpanderPNA(deg, args.num_readout_layers, args.density, args.sample_method, args.weight_initializer, args.dense_output).to(device)
    else:
        model = PNA(deg, args.num_readout_layers).to(device)
    
    # model = model.to(device)

    if args.use_expander:
        density = float(len(model.edge_index_list[0][0][0]) / (50*75))
    else:
        density = 1.0
        
    outdir = osp.join(osp.dirname(osp.realpath(__file__)), args.outdir, "{}-{}-{}".format(args.dataset, args.num_readout_layers, args.dense_output))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    file = "{}-{}-{}-{}-{}".format(density, args.use_expander, args.sample_method, args.weight_initializer, datetime.now().strftime("%m%d_%H%M%S"))
    logname = osp.join(outdir, "%s.log" % (file))

    log = logging.getLogger(file)
    log.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.FileHandler(filename=logname)  # output to file
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(handler)

    chler = logging.StreamHandler()  # print to console
    chler.setFormatter(logging.Formatter(fmt, datefmt))
    chler.setLevel(logging.INFO)
    log.addHandler(chler)

    log.info("Experiment of model: %s" % (file))
    log.info(args)   

    logger = Logger(args.runs, args, log)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)
        for epoch in range(1, 1 + args.epochs):
            start_time = time()
            loss = train(model, train_loader, optimizer, device)
            val_mae = test(model, val_loader, device)
            train_time = time()
            test_mae = test(model, test_loader, device)
            infer_time = time()
            scheduler.step(val_mae)
            logger.add_result(run, (-loss, -val_mae, -test_mae))

            if epoch % args.log_steps == 0:
                log.info(f'Run: {run + 1:02d}, '
                         f'Epoch: {epoch:02d}, '
                         f'Train Time: {train_time-start_time:.6f}, '
                         f'Infer Time: {infer_time-train_time:.6f}, '
                         f'Loss: {loss:.4f}, '
                         f'Val MAE: {val_mae:.4f} '
                         f'Test MAE: {test_mae:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()