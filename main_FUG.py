# GPL License
# Copyright (C) 2023 , UESTC
# All Rights Reserved
#
# @Time    : 2023/10/1 20:29
# @Author  : Qi Cao
import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Toolbox.data_FUG import Dataset
from Toolbox.losses import FUG_Losses
from Toolbox.model_FUG import FusionNet
from Toolbox.indexes import *
from Toolbox.model_SDE import Net_ms2pan
from Toolbox.wald_utilities import wald_protocol_v1


# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
# ============= HYPER PARAMS(Pre-Defined) ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--ckpt", type=int, default=1000, help="Checkpoint interval")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--device", type=str, default='cuda', help="Device to use")
parser.add_argument("--name", type=int, required=True, help="Data ID (0-19)")
parser.add_argument("--satellite", type=str, default='wv3/', help="Satellite type")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device)
name = args.name
satellite = args.satellite

model = FusionNet().to(device)
model.load_state_dict(torch.load(f'model_RSP/{satellite}{name}'))
criterion = FUG_Losses(device, f'model_SDE/{satellite}{name}_Net_ms2pan.pth')
F_ms2pan = Net_ms2pan().to(device)
F_ms2pan.load_state_dict(torch.load(f'model_SDE/{satellite}{name}_Net_ms2pan.pth'))
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))  # optimizer 1
betas = [8, 1]


# ============================================ #


def save_checkpoint(model, name):  # save model_FUG function
    model_out_path = 'model_FUG/' + satellite + name
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################


def train(training_data_loader, name):
    print('Run FUG...')
    min_loss = 100
    t1 = time.time()
    for epoch in range(epochs):
        epoch += 1
        epoch_train_loss1, epoch_train_loss2 = [], []
        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            ms, lms, pan = Variable(batch[0]).to(device), \
                           Variable(batch[1]).to(device), \
                           Variable(batch[2]).to(device)
            optimizer.zero_grad()  # fixed
            res = model(lms, pan)
            out = res + lms
            dsr = wald_protocol_v1(out, pan, 4, 'WV3')
            dpan = F_ms2pan(out)
            loss1, loss2 = criterion(out, pan, ms, dsr, dpan)  # compute loss

            total_loss = betas[0]*loss1+betas[1]*loss2
            epoch_train_loss1.append(loss1.item())  # save all losses into a vector for one epoch
            epoch_train_loss2.append(loss2.item())
            total_loss.backward()  # fixed
            optimizer.step()  # fixed

        t_loss1 = np.nanmean(np.array(epoch_train_loss1))  # compute the mean value of all losses, as one epoch loss
        t_loss2 = np.nanmean(np.array(epoch_train_loss2))
        t_total_loss = betas[0]*t_loss1 + betas[1]*t_loss2

        if epoch % 10 == 0:

            print('FUG stage: Epoch: {} [{:.7f} {:.7f}]  '
                  't_loss: {:.7f} '
                  .format(epoch, betas[0] * t_loss1, betas[1] * t_loss2, t_total_loss))
        if t_total_loss < min_loss:
            min_loss = t_total_loss
            save_checkpoint(model, name)
    t2 = time.time()
    print(f'FUG time: {t2-t1}s')

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################


if __name__ == "__main__":
    train_set = Dataset('dataset/' + satellite + 'train.h5', name)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    train(training_data_loader, str(name))