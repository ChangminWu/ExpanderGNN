import argparse
import os
import os.path as osp
import logging

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
from models import ExpanderSAGE, ExpanderGCN, SAGE, GCN


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='Node-Classification')
    parser.add_argument('--dataset', type=str, default="arxiv")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-steps', type=int, default=1)

    parser.add_argument('--use-expander', action='store_true')
    parser.add_argument('--use-sage', action='store_true')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hiddim', type=int, default=256)

    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--sample-method', type=str, default='prabhu')
    parser.add_argument('--weight-initializer', type=str, default='glorot')
    parser.add_argument('--dense-output', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    if args.weight_initializer == 'default':
        args.weight_initializer = None

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_expander:
        if args.use_sage:
            model = ExpanderSAGE(data.num_features, args.hiddim, dataset.num_classes, args.num_layers, args.dropout, args.density, args.sample_method, args.weight_initializer, args.dense_output).to(device)
        else:
            model = ExpanderGCN(data.num_features, args.hiddim, dataset.num_classes, args.num_layers, args.dropout, args.density, args.sample_method, args.weight_initializer, args.dense_output).to(device)
    else:
        if args.use_sage:
            model = SAGE(data.num_features, args.hiddim, dataset.num_classes, args.num_layers, args.dropout).to(device)
        else:
            model = GCN(data.num_features, args.hiddim, dataset.num_classes, args.num_layers, args.dropout).to(device)

    density = float(len(model.edge_index_list[1]) / (args.hiddim*args.hiddim))


    args.experiment = "{}-{}-{}-{}".format(args.exp_type, args.noise_type, args.merged, args.add_kernel)

    def name_folder_path():
        path = osp.join(args.out_dir, args.experiment, "{}-{}-{}".format(args.dataset, args.num_layers, args.readout))
        file = ("{}_"*15).format(args.hiddim, args.noise_ratio, args.epsilon,
                                 args.add_feat_noise, args.feat_noise_ratio, args.nystrom,
                                 args.random_noise_type, args.kernel_type,
                                 args.add_identity, args.normalize, args.spectral_norm, args.rf_norm,
                                 args.standarize, args.centerize, args.split_type)
        return path, file

    path, file = name_folder_path()
    path = osp.join(osp.dirname(osp.realpath(__file__)), path)

    log_dir = osp.join(path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    task_name = file + datetime.now().strftime("%m%d_%H%M%S")
    logname = osp.join(log_dir, "%s.log" % (task_name))

    log = logging.getLogger(task_name)
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

    log.info("Experiment of model: %s" % (task_name))
    log.info(args)


    

    if args.dataset == 'arxiv':
        evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()