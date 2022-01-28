class Evaluator(object):
    
    def __init__(self, metric, **kwargs):
        if metric == 'isomorphism':
            self.eval_fn = self._isomorphism
            self.eps = kwargs.get('eps', 0.01)
            self.p_norm = kwargs.get('p', 2)
        elif metric == 'accuracy':
            self.eval_fn = self._accuracy
        elif metric == 'mae':
            self.eval_fn = self._mae
        elif metric.startswith('ogb'):
            self._ogb_evaluator = OGBEvaluator(metric)
            self._key = self._ogb_evaluator.eval_metric
            self.eval_fn = self._ogb
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))
    
    def eval(self, input_dict):
        return self.eval_fn(input_dict)
        
    def _isomorphism(self, input_dict):
        # NB: here we return the failure percentage... the smaller the better!
        preds = input_dict['y_pred']
        assert preds is not None
        assert preds.dtype == np.float64
        preds = torch.tensor(preds, dtype=torch.float64)
        mm = torch.pdist(preds, p=self.p_norm)
        wrong = (mm < self.eps).sum().item()
        metric = wrong / mm.shape[0]
        return metric
    
    def _accuracy(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = np.argmax(input_dict['y_pred'], axis=1)
        assert y_true is not None
        assert y_pred is not None
        metric = met.accuracy_score(y_true, y_pred)
        return metric

    def _mae(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = met.mean_absolute_error(y_true, y_pred)
        return metric
    
    def _ogb(self, input_dict, **kwargs):
        assert 'y_true' in input_dict
        assert input_dict['y_true'] is not None
        assert 'y_pred' in input_dict
        assert input_dict['y_pred'] is not None
        return self._ogb_evaluator.eval(input_dict)[self._key]


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]