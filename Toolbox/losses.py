import torch
from torch import nn
from model_RSP import Net_ms2pan

class FUG_Losses(nn.Module):
    def __init__(self, device, ms2pan_pth_path):
        super(FUG_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)
        F_ms2pan = Net_ms2pan().to(device)
        F_ms2pan.load_state_dict(torch.load(ms2pan_pth_path))
        self.F_ms2pan = F_ms2pan

    def forward(self, sr, pan, ms, dsr, dpan):
        loss1 = self.mse(dsr, ms)
        loss2 = self.mse(dpan, pan)

        return loss1, loss2


class RSP_Losses(nn.Module):
    def __init__(self, device):
        super(RSP_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)

    def forward(self, out, pan):
        loss = self.mse(out, pan)

        return loss


class SDE_Losses(nn.Module):
    def __init__(self, device):
        super(SDE_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)

    def forward(self, lms_rr, pan_rr):
        loss = self.mse(lms_rr, pan_rr)

        return loss