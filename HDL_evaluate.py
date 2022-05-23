import os.path as osp
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
import time
from networks.deeplab import *
from datasets.cityscapes_dataset import cityscapesDataSet, cityscapesDataSetSYN
import timeit
import os
import torch
BACKBONE = 'resnet_hdl'
IGNORE_LABEL = 255
NUM_CLASSES = 19
LOG_DIR = './logs'
#DATA_DIRECTORY = '/path/to/cityscapes'
DATA_DIRECTORY = '/mnt/mdisk/zyh/DADatasets/Cityscapes'

DATA_LIST_PATH = './datasets/cityscapes_list/val.txt'
RESTORE_FROM = 'pretrained/'
EXP_ROOT = ""
CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall',
    'fence', 'pole', 'light', 'sign',
    'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train',
    'motocycle', 'bicycle'
]
DATA_LIST_PATH_TARGET = './datasets/cityscapes_list/train.txt'
DATA_DIRECTORY_TARGET = '/mnt/mdisk/zyh/DADatasets/Cityscapes'
PATH_PTH = ""
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help=".")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--img-width", type=int, default=1024)
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--list-path", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--is-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--split", type=str, default='val',
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--exp-root", type=str, default=EXP_ROOT,
                        help="Where restore experience.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--path_file", type=str, default=PATH_PTH,
                        help="model path.")
    return parser.parse_args()

def scale_image(image, scale):
    _, _, h, w = image.size()
    scale_h = int(h*scale)
    scale_w = int(w*scale)
    image = F.interpolate(image, size=(scale_h, scale_w),
                          mode='bilinear', align_corners=True)
    return image

def predictHDL(net, image, output_size, is_mirror=True, scales=[1]):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    outputs = []
    if is_mirror:
        # image_rev = image[:, :, :, ::-1]
        image_rev = torch.flip(image, dims=[3])
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image
                image_rev_scale = image_rev
            image_scale = torch.cat([image_scale, image_rev_scale], dim=0)  #  concat two images
            with torch.no_grad():
                prediction = net(image_scale.cuda(), src=2)
                prediction = interp(prediction)
            prediction_rev = prediction[1, :, :, :].unsqueeze(0)
            prediction_rev = torch.flip(prediction_rev, dims=[3])
            prediction = prediction[0, :, :, :].unsqueeze(0)
            prediction = (prediction + prediction_rev)*0.5  # 把两个结果加起来
            outputs.append(prediction)
        outputs = torch.cat(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)
        outputs = outputs.permute(1, 2, 0)
    else:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
            else:
                image_scale = image
            with torch.no_grad():
                prediction = net(image_scale.cuda(), src=2)
            prediction = interp(prediction)
            outputs.append(prediction)
        outputs = torch.cat(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)
        outputs = outputs.permute(1, 2, 0)
    probs, pred = torch.max(outputs, dim=2)
    pred = pred.cpu().data.numpy()
    return pred

def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred_label] = label_count[cur_index]

    return confusion_matrix

def display_stats(args, name_classes, inters_over_union_classes):
    for ind_class in range(args.num_classes):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))

def main_eval():
    """Create the model and start the evaluation process."""
    start = timeit.default_timer()
    args = get_arguments()
    pprint(vars(args))
    print("=======================================")
    print("Exp root:", args.exp_root)
    h, w = args.img_height, args.img_width
    if 'G2C' in args.path_file:
        test_loader = cityscapesDataSet(root=args.data_dir, list_path=args.list_path, set=args.split, img_size=(
            2048, 1024), norm=False, ignore_label=args.ignore_label)
    else:
        args.num_classes = 16
        print("classes :", args.num_classes)
        test_loader = cityscapesDataSetSYN(root=args.data_dir, list_path=args.list_path, set=args.split, img_size=(
            2048, 1024), norm=False, ignore_label=args.ignore_label)
    test_loader = DataLoader(test_loader, batch_size=1,
                             shuffle=False, num_workers=4)
    model = Deeplab_Res101HDL(num_classes=args.num_classes)
    pbar = tqdm(enumerate(test_loader))
    print("len loader:", len(test_loader))
    path_pth = args.path_file
    print(path_pth)
    if not osp.exists(path_pth):
        print('Waiting for model..!')
        while not osp.exists(path_pth):
            time.sleep(10)
    print("Evaluating model", path_pth)
    # load model ...
    saved_state_dict = torch.load(path_pth)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()
    test_iter = iter(test_loader)
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    confusion_cost = 0
    for index in tqdm(range(len(test_loader))):
        if index % 100 == 0:
            print('\n %d processd' % (index))
        image, label, size, name = next(test_iter)
        image = F.interpolate(image, size=(
            h, w), mode='bilinear', align_corners=True)
        pred = predictHDL(model, image.cuda(), (1024, 2048))
        seg_pred = np.asarray(pred, dtype=np.uint8)  #
        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_start = timeit.default_timer()
        confusion_matrix += get_confusion_matrix(
            seg_gt, seg_pred, args.num_classes)
        confusion_cost += timeit.default_timer() - confusion_start
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    end = timeit.default_timer()
    print("Total time:", end-start, 'seconds')
    mean_IU= round(np.nanmean(IU_array) * 100, 2)
    print('\tCurrent Mean IU: %s' % str(mean_IU))
    display_stats(args, CLASS_NAMES, IU_array)
    print("\n")

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    main_eval()
