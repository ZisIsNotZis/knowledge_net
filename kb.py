import torch
from torch.nn import *

torch.set_default_tensor_type(torch.cuda.FloatTensor)


def flat(x):
    if isinstance(x, list):
        for x in x:
            yield from flat(x)
    else:
        yield x


class F(Module):
    def __init__(s, f):
        super().__init__()
        s.f = f

    def forward(s, x):
        return s.f(x)


class Lift(Sequential):
    def __init__(s, n, *m):
        super().__init__(*m)
        s.n = n

    def forward(s, x):
        t = super().forward(x.view(-1, *x.shape[-s.n :]))
        return t.view(*x.shape[: -s.n], *t.shape[1:])


class Linears(Linear):
    def forward(s, x):
        x = list(flat(x))
        if len(x) < 2:
            return super().forward(x[0])
        w = s.weight.split([x.size(-1) for x in x], 1)
        return sum([functional.linear(x, w) for x, w in zip(x[1:], w[1:])], functional.linear(x[0], w[0], s.bias))


class Nn(Module):
    def __init__(s, i, h, o=0, inplace=1):
        super().__init__()
        s.nn = Sequential(Linears(i, h, 0), Lift(1, BatchNorm1d(h, track_running_stats=0)), Hardswish(inplace), *[Linear(h, o)] if o else [])

    def forward(s, x):
        return s.nn(x)

    def skip(s, x):
        return s.nn[1:](x)


class Notice(Module):
    def __init__(s, q, k, h):
        super().__init__()
        s.nn = Nn(q + k + 1, h, 1)

    def forward(s, q, k):
        return s.nn([q, k, torch.linspace(0, 1, k.size(-2))[:, None]]).softmax(-2)


class Bitention(Module):
    def __init__(s, i, h):
        super().__init__()
        s.read = Notice(i, h, h)
        s.readNn = Nn(h, h)
        s.write = Notice(h, h, h)
        s.nn = Nn(i + h, h + h)

    def forward(s, i, h):
        o, i = s.nn([i, s.read(i, h)[..., None, :, 0] @ s.readNn(h)]).chunk(2, -1)
        t = s.write(i, h)
        return o, t * i + (1 - t) * h


class Kb(Module):
    def __init__(s, i, h, l):
        super().__init__()
        s.nn = Bitention(i, h)
        s.l = h
        s.h = h

    def forward(s, x):
        h = torch.empty(s.l, s.h)
        o = []
        for x in x:
            x, h = s.nn(x, h)
            o.append(x)
        return torch.stack(o)