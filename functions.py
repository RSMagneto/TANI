import torch
import numpy as np
import pdb
from sewar.full_ref import sam


def matRead(data,opt):
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    return data

def test_matRead(data,opt):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    return data

def getBatch(ms_data,pan_data,gt_data, bs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    # pdb.set_trace()
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def getTest(ms_data,pan_data,gt_data,bs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def convert_image_np(inp,opt):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    inp=inp*2047.
    return inp

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img,hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.numpy()
    sr_img=sr_img[-1,:,:,:]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value

def SID(MS,PANMS,opt):
    b,d,n,m=PANMS.shape
    p=torch.zeros_like(PANMS)
    q=torch.zeros_like(PANMS)
    for i in range(d):
        p[:,i,:,:]=(MS[:,i,:,:])/torch.sum(MS,dim=1)
        q[:,i,:,:]=(PANMS[:,i,:,:])/torch.sum(PANMS,dim=1)
    S=torch.zeros([b,n,m],device=opt.device)
    N=torch.zeros([b,n,m],device=opt.device)
    for i in range(d):
        S=(p[:,i,:,:]*torch.log(p[:,i,:,:]/q[:,i,:,:]))+S
        N = (q[:, i, :, :] * torch.log(q[:, i, :, :] / p[:, i, :, :])) + N
    D=N+S
    sumD=torch.sum(D)/(n*m)
    return sumD