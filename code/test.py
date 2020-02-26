import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from scipy.spatial.distance import sqeuclidean
from librosa.sequence import dtw as librosa_dtw
from dtw import dtw as py_dtw
from dtwco import dtw as dtwco
from _ext import dynamic_prog_lib

L1 = random.randint(5, 10)
L2 = random.randint(5, 10)
A = torch.rand(L1, requires_grad=True)
B = torch.rand(L2, requires_grad=True)


def test_dtw():
    print("Aligning with DTWNet forward method")
    dtwpath = torch.zeros((L1 + L2, 2), dtype=torch.int32)
    # sqeuclidean is hard-coded
    path_len = dynamic_prog_lib.dynamic_prog_lib_forward(A, B, dtwpath)
    dtwpath = dtwpath[:path_len].numpy()
    print("    len(dtwnet): ", path_len)

    a = A.numpy()
    b = B.numpy()

    print("Aligning with librosa")
    cost, path = librosa_dtw(a, b, metric='sqeuclidean')
    print("    len(librosa): ", path.shape[0])
    print("    average diff: ", np.mean(np.abs(path - dtwpath)))

    print("Aligning with dtw")
    _d, _cost_matrix, _acc_cost_matrix, path = py_dtw(a, b, dist=sqeuclidean)
    print("    len(dtw): ", len(path[0]))
    path = np.array(path).T[::-1]
    print("    average diff: ", np.mean(np.abs(path - dtwpath)))

    print("Aligning with dtwco")
    _d, _c, path = dtwco(a, b, metric='sqeuclidean', dist_only=False)
    print("    len(dtw): ", len(path[0]))
    path = np.array(path).T[::-1]
    print("    average diff: ", np.mean(np.abs(path - dtwpath)))


class DTW(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(nn.Linear(L1, L1), )
        self.seq2 = nn.Sequential(nn.Linear(L2, L2), )

    def forward(self, a, b):
        a = self.seq1(a)
        b = self.seq2(b)
        dtwpath = torch.zeros((L1 + L2, 2),
                              dtype=torch.int32,
                              requires_grad=True)
        dtwpath.register_hook(lambda x: print(x))
        a.register_hook(lambda x: print(x))
        b.register_hook(lambda x: print(x))
        for p in self.seq1.parameters():
            p.register_hook(lambda x: print(x))
        for p in self.seq2.parameters():
            p.register_hook(lambda x: print(x))

        dtwpath.retain_grad()
        # sqeuclidean is hard-coded
        path_len = dynamic_prog_lib.dynamic_prog_lib_forward(a, b, dtwpath)

        # using the following it computes the DTW distance as in DTWNet
        # return (sum([(a[dtwpath[i][0]] - b[dtwpath[i][1]])**2
        #              for i in range(path_len)]))
        return dtwpath[:path_len]


def test_backprop():
    model = DTW()
    optim = torch.optim.Adam(model.parameters())
    out = model(A, B).to(torch.float)
    target = torch.randint_like(out, 0, out.shape[0]).to(torch.float)

    # use the following for testing DTW distance as in DTWNet (see also DTW
    # model above)
    # target = torch.randint_like(out, 0, 1).to(torch.float)

    loss = F.l1_loss(out, target)

    print("Out: ", out)
    print("Target: ", target)
    print("Loss: ", loss)

    loss.backward()

    print(A.grad, B.grad, out.grad, loss.grad)

    optim.step()
    print("Finished!")


if __name__ == '__main__':
    import sys
    eval(sys.argv[1])()
