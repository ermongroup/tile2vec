import numpy as np
from time import time
from torch.autograd import Variable
from src.datasets import triplet_dataloader

def prep_triplets(triplets, cuda):
    """
    Takes a batch of triplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    a, n, d = (Variable(triplets['anchor']), Variable(triplets['neighbor']), Variable(triplets['distant']))
    if cuda:
    	a, n, d = (a.cuda(), n.cuda(), d.cuda())
    return (a, n, d)

def train_triplet_epoch(model, cuda, dataloader, optimizer, epoch, margin=1,
    l2=0, print_every=100, t0=None):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    model.train()
    if t0 is None:
        t0 = time.time()
    sum_loss, sum_l_n, sum_l_d, sum_l_nd = (0, 0, 0, 0)
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    for idx, triplets in enumerate(dataloader):
        p, n, d = prep_triplets(triplets, cuda)
        optimizer.zero_grad()
        loss, l_n, l_d, l_nd = model.loss(p, n, d, margin=margin, l2=l2)
        loss.backward()
        optimizer.step()
        sum_loss += loss.data[0]
        sum_l_n += l_n.data[0]
        sum_l_d += l_d.data[0]
        sum_l_nd += l_nd.data[0]
        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
    avg_loss = sum_loss / n_batches
    avg_l_n = sum_l_n / n_batches
    avg_l_d = sum_l_d / n_batches
    avg_l_nd = sum_l_nd / n_batches
    print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
    print('  Average loss: {:0.4f}'.format(avg_loss))
    print('  Average l_n: {:0.4f}'.format(avg_l_n))
    print('  Average l_d: {:0.4f}'.format(avg_l_d))
    print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))
    return (avg_loss, avg_l_n, avg_l_d, avg_l_nd)
