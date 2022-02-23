import argparse
import model as model
import torch
import torch.nn as nn
import functions
import time
import os
import copy
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', required=True)#, default='')
    parser.add_argument('--test_dir', help='testing_data', required=True)#, default='')
    parser.add_argument('--outputs_dir',help='output model dir', required=True)#, default='')
    parser.add_argument('--channels',help='numble of image channel', default=4)
    parser.add_argument('--batchSize', default=32)
    parser.add_argument('--testBatchSize', default=1)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device',default=torch.device('cuda'))
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr',type=float,default=0.0001,help='G‘s learning rate')
    parser.add_argument('--gamma',type=float,default=0.01,help='scheduler gamma')
    parser.add_argument('--lr_decay_step', type=int, default=250)
    parser.add_argument('--lr_decay_rate', type=float,default=0.95)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    train_set = dataloader.get_training_set(opt.input_dir)
    val_set = dataloader.get_val_set(opt.test_dir)

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    # 网络初始化：
    net = model.Net(opt,spatial_height=64, spatial_width=64).to(opt.device)
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    # 建立优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[1600],gamma=opt.gamma)

    # loss=torch.nn.MSELoss()
    loss = torch.nn.L1Loss()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        net = net.cuda()
        loss = loss.cuda()
    best_weights = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_SAM=1.0
    for i in range(opt.epoch):
        # train
        net.train()
        epoch_losses = functions.AverageMeter()
        batch_time = functions.AverageMeter()
        end = time.time()
        for batch_idx, (gtBatch, msBatch, panBatch) in enumerate(train_loader):

            if torch.cuda.is_available():
                msBatch, panBatch, gtBatch = msBatch.cuda(), panBatch.cuda(), gtBatch.cuda()
                msBatch = Variable(msBatch.to(torch.float32))
                panBatch = Variable(panBatch.to(torch.float32))
                gtBatch = Variable(gtBatch.to(torch.float32))
            msBatch,panBatch,gtBatch=functions.getBatch(msBatch,panBatch,gtBatch,opt.batchSize)
            N = len(train_loader)
            net.zero_grad()
            msBatch = torch.nn.functional.interpolate(msBatch, size=(gtBatch.shape[2], gtBatch.shape[3]),
                                                      mode='bilinear')
            mp=net(msBatch,panBatch)
            mseLoss=loss(mp,gtBatch)
            mseLoss.backward(retain_graph=True)
            sidLoss=0.1*functions.SID(gtBatch,mp,opt)
            if sidLoss==sidLoss:
                sidLoss.backward()
            optimizer.step()
            epoch_losses.update(mseLoss.item(), msBatch.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_idx + 1) % 100 == 0:
                training_state = '  '.join(
                    ['Epoch: {}', '[{} / {}]', 'mseLoss: {:.6f}','sidLoss: {:.6f}']
                )
                training_state = training_state.format(
                    i, batch_idx, N, mseLoss, sidLoss
                )
                print(training_state)

        print('%d epoch: loss is %.6f, epoch time is %.4f' % (i, epoch_losses.avg, batch_time.avg))
        torch.save(net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(i)))
        net.eval()
        epoch_SAM=functions.AverageMeter()
        with torch.no_grad():
            for j, (gtTest, msTest, panTest) in enumerate(val_loader):
                if torch.cuda.is_available():
                    msTest, panTest, gtTest = msTest.cuda(), panTest.cuda(), gtTest.cuda()
                    msTest = Variable(msTest.to(torch.float32))
                    panTest = Variable(panTest.to(torch.float32))
                    gtTest = Variable(gtTest.to(torch.float32))
                    net = net.cuda()
                msTest = torch.nn.functional.interpolate(msTest, size=(256, 256), mode='bilinear')
                mp = net(msTest, panTest)
                test_SAM=functions.SAM(mp, gtTest)
                if test_SAM==test_SAM:
                    epoch_SAM.update(test_SAM,msTest.shape[0])
            print('eval SAM: {:.6f}'.format(epoch_SAM.avg))

        if epoch_SAM.avg < best_SAM:
            best_epoch = i
            best_SAM = epoch_SAM.avg
            best_weights = copy.deepcopy(net.state_dict())
        print('best epoch:{:.0f}'.format(best_epoch))
        scheduler.step()

    print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))
