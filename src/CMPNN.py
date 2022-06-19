import torch as th
from torch import nn
import dgl.function as fn


class CMPNNConv(nn.Module):
    def __init__(self,
                 node_feats,
                 edge_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 activation=None,
                 ):
        super(CMPNNConv, self).__init__()
        self._out_feats = out_feats
        self._k = k
        self._activation = activation
        self.lin = nn.Linear(edge_feats*2+node_feats, out_feats, bias=bias)


        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def message(self, edge):
        msg = edge.src['h']
        return {'e': msg}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (node_feat.dim() - 1)
            norm = th.reshape(norm, shp).to(node_feat.device)

            msg_func = fn.copy_e("e", "m")
            graph.edata['e'] = edge_feat
            fstack = [node_feat] 
            for _ in range(self._k):
                rst = fstack[-1] * norm
                graph.ndata['h'] = rst

                graph.update_all(msg_func,
                                 fn.sum(msg='m', out='h'))
                rst = graph.ndata['h']
                rst = rst * norm
                graph.apply_edges(self.message)
                fstack.append(rst)

            rst = self.lin(th.cat(fstack, dim=-1))
            graph.ndata['h'] = rst
            graph.apply_edges(self.message)
            est = graph.edata['e']

            if self._activation is not None:
                rst = self._activation(rst)
                est = self._activation(est)
            return rst, est