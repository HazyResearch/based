from torch import nn

def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation in ['sqrelu', 'relu2']:
        return SquaredReLU()
    elif activation == 'laplace':
        return Laplace()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
