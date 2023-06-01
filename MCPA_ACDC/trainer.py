import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F


def trainer_ACDC(args, k, config, model, snapshot_path):
    # print('k',k)

    best_dir = os.path.join(snapshot_path, config.NAME, str(k))
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    txt_dir = os.path.join(snapshot_path, config.NAME, str(k), 'log.txt')
    with open(txt_dir, 'a', encoding='utf-8') as f:
        f.write('1')
    from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/" + config.NAME + '/' + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(config)
    logging.info(str(args))
    base_lr = config.LR
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
                            norm_x_transform=x_transforms, norm_y_transform=y_transforms)

    # print(args.list_dir)
    print("The length of train set is: {}".format(len(db_train)))

    # -------------------------

    # -----------------------
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # print("trainloader.datasetlen",len(trainloader.dataset))
    # for  sampled_batch in enumerate(trainloader):
    # print("i",i_batch)
    #   print("sampled_batch",sampled_batch)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(4)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, config.NAME, str(k)))
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    print("ready!!")

    for epoch_num in iterator:
        print('epoch_num', epoch_num)

        for i_batch, sampled_batch in enumerate(trainloader):
            # print("image_batch", i_batch)
            # print("label_batch", sampled_batch)

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # image_batch = image_batch.float()
            # label_batch = label_batch.float()

            # print(image_batch.dtype)
            # print(label_batch.dtype)

            # torch.Tensor(image_batch)
            # torch.Tensor(label_batch)

            # print("ok")
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            # torch.Tensor(image_batch)

            outputs = model(image_batch)
            # outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            loss_ce = ce_loss(outputs, label_batch[:].long())

            loss_dice = dice_loss(outputs, label_batch, softmax=False)

            loss = 0.4 * loss_ce + 0.6 * loss_dice
            # print("loss-----------", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)

        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, config.NAME,str(k),'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, config.NAME, str(k), 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_synapse(args, k, config, model, snapshot_path):
    best_dir = os.path.join(snapshot_path, config.NAME, str(k))
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    txt_dir = os.path.join(snapshot_path, config.NAME, str(k), 'log.txt')
    with open(txt_dir, 'a', encoding='utf-8') as f:
        f.write('1')
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/" + config.NAME + '/' + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(config)
    logging.info(str(args))
    base_lr = config.LR
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
                               norm_x_transform=x_transforms, norm_y_transform=y_transforms)

    print("The length of train set is: {}".format(len(db_train)))

    # -------------------------

    # -----------------------
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(4)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, config.NAME, str(k)))
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            outputs = model(image_batch)
            # outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            loss_ce = ce_loss(outputs, label_batch[:].long())

            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss = 0.4 * loss_ce + 0.6 * loss_dice
            # print("loss-----------", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)

        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, config.NAME,str(k),'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, config.NAME, str(k), 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
