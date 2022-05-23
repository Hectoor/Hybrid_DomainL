from pprint import pprint
import timeit
import torch.optim as optim
import torch.backends.cudnn as cudnn
from networks.deeplab import *
from networks.discriminator import *
from utils.loss import CrossEntropy2d
from datasets.gta5_dataset import GTA5DataSet
from datasets.cityscapes_dataset import *
from options import gta5asa_opt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import torch
from torch.autograd import Variable
args = gta5asa_opt.get_arguments()

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda()
    criterion = CrossEntropy2d().cuda()
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main_HDL():
    """Create the model and start the training."""
    save_dir = osp.join(args.snapshot_dir, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    cudnn.enabled = True
    model = Deeplab_Res101HDL(num_classes=args.num_classes)
    RESTORE_FROM = args.restore_from
    assert osp.exists(RESTORE_FROM), f'Missing init model {RESTORE_FROM}'
    if 'GTA5_init' in RESTORE_FROM:
        print('train from scatch ...')
        saved_state_dict = torch.load(RESTORE_FROM)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 19 or not i_parts[0] == 'layer5' and not i_parts[0] == 'fc':
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:
        print("Resuming from ==>>", RESTORE_FROM)
        state_dict = torch.load(RESTORE_FROM)
        model.load_state_dict(state_dict)
    model.train()
    model.cuda()
    cudnn.benchmark = True
    Iter = 0
    pprint(vars(args))
    model_D = EightwayASADiscriminator(num_classes=args.num_classes)
    if args.continue_train:
        model_weights_path = args.restore_from
        temp = model_weights_path.split('.')
        temp[-2] = temp[-2] + '_D'
        model_D_weights_path = '.'.join(temp)
        print(model_D_weights_path)
        model_D.load_state_dict(torch.load(model_D_weights_path))
        temp = model_weights_path.split('.')
        temp = temp[-2][-9:]
        Iter = int(temp.split('_')[1]) + 1
        print("load model D")
    model_D.train()
    model_D.cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    img_size=input_size),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.batch_size,
                                                     img_size=input_size_target,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)
    test_loader = cityscapesDataSet(root=args.data_dir_target, list_path=args.data_list_path_val, set='val', img_size=(
        2048, 1024), norm=False, ignore_label=args.ignore_label)
    test_loader = DataLoader(test_loader, batch_size=1,
                             shuffle=False, num_workers=4)
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    optimizer_D = optim.Adam(model_D.parameters(
    ), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    interp = nn.Upsample(
        size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(
        input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    source_label = 0
    target_label = 1
    start = timeit.default_timer()
    loss_seg_value = 0
    loss_adv_target_value = 0
    loss_D_value = 0
    for i_iter in range(Iter, args.num_steps+1):
        damping = (1 - i_iter/args.num_steps)
        lambda_trg = 0.1
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)
        for param in model_D.parameters():
            param.requires_grad = False
        # train with target
        _, batch = next(targetloader_iter)
        tar_img, _, _, _ = batch
        tar_img = Variable(tar_img).cuda()
        with torch.no_grad():
            feat_ = model(tar_img, feat_=None, src=0)
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        # train with source
        _, batch = next(trainloader_iter)
        src_img, labels, _, _ = batch
        src_img = Variable(src_img).cuda()
        pred = model(src_img, feat_, src=1, lambda_trg=lambda_trg)    # src=1
        pred = interp(pred)
        loss_seg = loss_calc(pred, labels)
        loss_seg.backward()
        loss_seg_value += loss_seg.item()

        # train with target
        pred_target = model(tar_img, feat_=1, src=0, lambda_trg=lambda_trg)
        pred_target = interp_target(pred_target)
        D_out = model_D(F.softmax(pred_target, dim=1))
        loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
        loss_adv = loss_adv_target * args.lambda_adv_target1 * damping
        loss_adv_target_value += loss_adv_target.item()
        loss_adv.backward()

        for param in model_D.parameters():
            param.requires_grad = True
        # train with source
        pred = pred.detach()
        D_out = model_D(F.softmax(pred, dim=1))
        loss_D1 = bce_loss(D_out, torch.FloatTensor(
            D_out.data.size()).fill_(source_label).cuda())
        loss_D1 = loss_D1 / 2
        loss_D1.backward()

        loss_D_value += loss_D1.item()
        # train with target
        pred_target = pred_target.detach()
        D_out1 = model_D(F.softmax(pred_target, dim=1))
        loss_D1 = bce_loss(D_out1, torch.FloatTensor(
            D_out1.data.size()).fill_(target_label).cuda())
        loss_D1 = loss_D1 / 2
        loss_D1.backward()
        loss_D_value += loss_D1.item()
        optimizer.step()
        optimizer_D.step()
        current = timeit.default_timer()

        if i_iter % 1000 == 0:
            print(
                'iter = {0:6d}/{1:6d}, loss_seg1 = {2:.3f}  loss_adv1 = {3:.3f}, loss_D1 = {4:.3f} ({5:.3f}/iter)'.format(
                    i_iter, args.num_steps, loss_seg_value/50,  loss_adv_target_value/50, loss_D_value/50, (current - start) / (i_iter+1))
            )
            print("dir :", args.snapshot_dir)
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalars("Loss", {
                               "Seg": loss_seg_value, "Adv": loss_adv_target_value, "Disc": loss_D_value}, i_iter)
            loss_seg_value = 0
            loss_adv_target_value = 0
            loss_D_value = 0
        if (i_iter % args.save_pred_every == 0 and i_iter != 0):
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(
                save_dir, 'HDLG2C' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(
                save_dir, 'HDLG2C' + str(i_iter) + '_D.pth'))
            if i_iter >= args.num_steps_stop:
                break

def main_HDS():
    """Create the model and start the training."""
    save_dir = osp.join(args.snapshot_dir, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    cudnn.enabled = True

    model = Deeplab_Res101HDL(num_classes=args.num_classes)

    RESTORE_FROM = args.restore_from
    assert osp.exists(RESTORE_FROM), f'Missing init model {RESTORE_FROM}'
    if 'GTA5_init' in RESTORE_FROM or 'resnet' in RESTORE_FROM:
        print('train from scatch ...')
        saved_state_dict = torch.load(RESTORE_FROM)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 19 or not i_parts[1] == 'layer5' and not i_parts[0] == 'fc':
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:
        print("Resuming from ==>>", RESTORE_FROM)
        state_dict = torch.load(RESTORE_FROM)
        model.load_state_dict(state_dict)

    model.train()
    model.cuda()
    cudnn.benchmark = True
    Iter = 0

    pprint(vars(args))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    img_size=input_size),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.batch_size,
                                                     img_size=input_size_target,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)
    test_loader = cityscapesDataSet(root=args.data_dir_target, list_path=args.data_list_path_val, set='val', img_size=(
        2048, 1024), norm=False, ignore_label=args.ignore_label)
    test_loader = DataLoader(test_loader, batch_size=1,
                             shuffle=False, num_workers=4)
    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(
        size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(
        input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    start = timeit.default_timer()
    loss_seg_value = 0
    if args.continue_train:
        model_weights_path = args.restore_from
        temp = model_weights_path.split('.')
        temp[-2] = temp[-2] + '_D'
        temp = model_weights_path.split('.')
        temp = temp[-2][-9:]
        Iter = int(temp.split('_')[1]) + 1
    for i_iter in range(Iter, args.num_steps+1):

        lambda_trg = 0.1
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        # get the target domain feature.
        _, batch = next(targetloader_iter)
        tar_img, _, _, _ = batch
        tar_img = Variable(tar_img).cuda()
        with torch.no_grad():
            feat_ = model(tar_img, feat_=None, src=0)
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # train with source
        _, batch = next(trainloader_iter)
        src_img, labels, _, _ = batch
        src_img = Variable(src_img).cuda()
        pred = model(src_img, feat_, src=1, lambda_trg=lambda_trg)    # src=1
        pred = interp(pred)
        loss_seg = loss_calc(pred, labels)
        loss_seg.backward()
        loss_seg_value += loss_seg.item()
        optimizer.step()
        current = timeit.default_timer()

        if i_iter % 1000 == 0:
            print(
                'iter = {0:6d}/{1:6d}, loss_seg1 = {2:.3f}, ({3:.3f}/iter)'.format(
                    i_iter, args.num_steps, loss_seg_value/50,  (current - start) / (i_iter+1))
            )
            print("only hdsloss", args.snapshot_dir)
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalars("Loss", {
                               "Seg": loss_seg_value}, i_iter)
            loss_seg_value = 0

        if (i_iter % args.save_pred_every == 0 and i_iter != 0):
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(
                save_dir, 'HDSG2C' + str(i_iter) + '.pth'))
            if i_iter >= args.num_steps_stop:
                break

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
if __name__ == '__main__':
    #main_HDL()
    main_HDS()
