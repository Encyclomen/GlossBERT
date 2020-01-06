from torch import nn
from torch.nn import init


def xavier_init(self):
    for m in self.modules():
        print(m)
        if isinstance(m, nn.Linear):
            # print(m.weight.data.type())
            # input()
            # m.weight.data.fill_(1.0)
            init.xavier_uniform_(m.weight, gain=1)
            print(m.weight)