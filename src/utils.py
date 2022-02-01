import os
import 

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


class LoadDataset(object):
    def __init__(self, root, name, pre_transform=None, transform=None):
        self.root = root
        self.name = name.lower()
        self.pre_transform = pre_transform
        self.transform = transform

    def load(self):
        if self.name == 'cora':
            self.dataset = Planetoid(root=os.path.join(self.root, 'datasets', self.name), name='Cora', )
        elif self.name == 'citeseer':
            self.dataset = Planetoid(root=os.path.join(self.root, 'datasets', self.name), name='CiteSeer', )

        else:
            raise NotImplementedError(self.name)