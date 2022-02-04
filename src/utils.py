import os
# import 

from ogb.nodeproppred import Evaluator as OGBNEvaluator
from ogb.graphproppred import Evaluator as OGBGEvaluator
from sklearn import metrics

from torch_geometric.datasets import Planetoid, ZINC


class Evaluator(object):
    def __init__(self, metric):
        if metric.startswith('ogbn'):
            self._evaluator = OGBNEvaluator(metric)
            self._key = 'acc' #self._evaluator.eval_metric
            self.eval_fn = self._ogb
        elif metric.startswith('ogbg'):
            self._evaluator = OGBGEvaluator(metric)
            self._key = 'acc'
            self.eval_fn = self._ogb
        elif metric == 'mae':
            self.eval_fn = self._mae
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))

    def eval(self, input_dict):
        return self.eval_fn(input_dict)

    def _ogb(self, input_dict, **kwargs):
        assert 'y_true' in input_dict
        assert input_dict['y_true'] is not None
        assert 'y_pred' in input_dict
        assert input_dict['y_pred'] is not None
        return self._evaluator.eval(input_dict)[self._key]

    def _mae(self, input_dict):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = metrics.mean_absolute_error(y_true, y_pred)
        return metric
    
    def _acc(self, input_dict):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = metrics.accuracy_score(y_true, y_pred)
        return metric


# class LoadDataset(object):
#     def __init__(self, root, name, pre_transform=None, transform=None):
#         self.root = root
#         self.name = name.lower()
#         self.pre_transform = pre_transform
#         self.transform = transform

#     def load(self):
#         if self.name == 'cora':
#             self.dataset = Planetoid(root=os.path.join(self.root, 'datasets', self.name), name='Cora', )
#         elif self.name == 'citeseer':
#             self.dataset = Planetoid(root=os.path.join(self.root, 'datasets', self.name), name='CiteSeer', )
#         else:
#             raise NotImplementedError(self.name)
    
#     def _ogb(self, input_dict, **kwargs):
#         assert 'y_true' in input_dict
#         assert input_dict['y_true'] is not None
#         assert 'y_pred' in input_dict
#         assert input_dict['y_pred'] is not None
#         return self._ogb_evaluator.eval(input_dict)[self._key]


# def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):

#     fill_value = 2. if improved else 1.

#     if isinstance(edge_index, SparseTensor):
#         adj_t = edge_index
#         if not adj_t.has_value():
#             adj_t = adj_t.fill_value(1., dtype=dtype)
#         if add_self_loops:
#             adj_t = fill_diag(adj_t, fill_value)
#         deg = sparsesum(adj_t, dim=1)
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
#         adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
#         return adj_t

#     else:
#         num_nodes = maybe_num_nodes(edge_index, num_nodes)

#         if edge_weight is None:
#             edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
#                                      device=edge_index.device)

#         if add_self_loops:
#             edge_index, tmp_edge_weight = add_remaining_self_loops(
#                 edge_index, edge_weight, fill_value, num_nodes)
#             assert tmp_edge_weight is not None
#             edge_weight = tmp_edge_weight

#         row, col = edge_index[0], edge_index[1]
#         deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#         return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
