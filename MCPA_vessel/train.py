import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys, time
from os.path import join
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from test import Test
from function import get_dataloader, train, val, get_dataloaderV2
from models.MCPA_MODEL.net import MCPA
from models.MCPA_MODEL.utils import *


def Criterion(outputs, outputs2, targets, device,epoch,args):
    ce_loss = CrossEntropyLoss2d().to(device)
    l1_loss = nn.SmoothL1Loss().to(device)
    dice_loss = DiceLoss().to(device)

    ce = ce_loss(outputs, targets)

    if epoch <=args.epoch_min:
        now_power=0
    elif epoch < args.epoch_max:
        now_power = ((epoch-args.epoch_min) /(args.epoch_max -args.epoch_min)) **args.alpha
    else:
        now_power = 1

    if args.epoch_min==args.epoch_max:
        now_power =1
    if outputs2 == [0]:
        loss = ce
    else:
        li = l1_loss(outputs, outputs2)
        tar = targets.unsqueeze(1)
        dice = dice_loss(outputs, tar.long())
        loss = 0.4 * ce + 0.6 * dice + li * now_power
        # print(now_power)

    return loss


def main():
    setpu_seed(2021)
    args = parse_args()
    strGPUs = [str(x) for x in [0, 1]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(strGPUs)
    device = torch.device("cuda")
    for k in range(200):
        save_path = join(args.outf, args.name, args.save, str(k))
        save_args(args, save_path)

        # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        # cudnn.benchmark = True

        log = Logger(save_path)
        sys.stdout = Print_Logger(os.path.join(save_path, 'train_log.txt'))
        print('The computing device used is: ', 'GPU' if device.type == 'cuda' else 'CPU')

        # net = models.UNetFamily.U_Net(1,2).to(device)
        # net = models.LadderNet(inplanes=args.in_channels, num_classes=args.classes, layers=4, filters=16) # default layers=3

        net = MCPA(num_classes=args.classes)
        net = torch.nn.DataParallel(net).to(device)
        print("Total number of parameters: " + str(count_parameters(net)))

        # # Load checkpoint.
        if args.pre_trained is not None:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        # create a list of learning rate with epochs
        # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)

        train_loader, val_loader = get_dataloaderV2(args)  # create dataloader

        if args.val_on_test:
            print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
            val_tool = Test(args)

        best = {'epoch': 0, 'AUC_roc': 0.5, 'F1': 0.5}  # Initialize the best epoch and performance(AUC of ROC)
        trigger = 0  # Early stop Counter
        for epoch in range(args.start_epoch, args.N_epochs + 1):
            print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
                  (epoch, args.N_epochs, optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))

            # train stage
            train_log = train(train_loader, net, Criterion, optimizer, device, epoch, args)
            # val stage
            if not args.val_on_test:
                val_log = val(val_loader, net, Criterion, device, epoch, args)
            else:
                print("start to infer on test")
                val_tool.inference(net)
                val_log = val_tool.val()

            print('update log')
            log.update(epoch, train_log, val_log)  # Add log information
            lr_scheduler.step()

            # Save checkpoint of latest and best model.
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, join(save_path, 'latest_model.pth'))
            trigger += 1
            if val_log['val_f1'] > best['F1']:
                print('update model.pth')
                print('\033[0;33mSaving best model!\033[0m')
                torch.save(state, join(save_path, 'best_model.pth'))
                best['epoch'] = epoch
                best['F1'] = val_log['val_f1']
                trigger = 0
            print('Best performance at Epoch: {} | f1: {}'.format(best['epoch'], best['F1']))
            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    print("=> early stopping")
                    break
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
