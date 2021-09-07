# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from functools import partial

import cv2
import mmcv
import numpy as np
from build.lib.mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of LSVT')
    parser.add_argument(
        'root_path', type=str, help='The path to the directory of LSVT')
    parser.add_argument(
        'n_proc', default=1, type=int, help='Number of processes to run')
    args = parser.parse_args()
    return args


def process_info(raw_ann_pair, src_img_root, dst_img_root):
    img_name, img_anns = raw_ann_pair
    img_path = osp.join(src_img_root, f'{img_name}.jpg')
    src_img = mmcv.imread(img_path, 'unchanged')

    labels = []
    for ann_idx, ann in enumerate(img_anns):
        if ann['illegibility']:
            continue
        text_label = ann['transcription']
        x, y, w, h = cv2.boundingRect(np.array(ann['points']))
        dst_img = src_img[y:y + h, x:x + w]
        dst_img_name = f'{img_name}_{ann_idx}.jpg'
        dst_img_path = osp.join(dst_img_root, dst_img_name)
        mmcv.imwrite(dst_img, dst_img_path)
        labels.append(f'{osp.basename(dst_img_root)}/{dst_img_name}'
                      f' {text_label}')
    return labels


def convert_lsvt(root_path,
                 dst_image_path,
                 dst_label_filename,
                 annotation_filename,
                 nproc=1):

    annotation_path = osp.join(root_path, annotation_filename)
    dst_label_file = osp.join(root_path, dst_label_filename)
    dst_image_root = osp.join(root_path, dst_image_path)
    os.makedirs(dst_image_root, exist_ok=True)

    img_root = osp.join(root_path, 'train_full_images')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} does not exist, please check and try again.')

    annotation = mmcv.load(annotation_path)

    process_info_with_img_root = partial(
        process_info, src_img_root=img_root, dst_img_root=dst_image_root)
    labels_list = mmcv.track_parallel_progress(
        process_info_with_img_root,
        annotation.items(),
        keep_order=True,
        nproc=nproc)
    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    list_to_file(dst_label_file, final_labels)


def main():
    args = parse_args()
    print('Processing training set...')
    convert_lsvt(
        root_path=args.root_path,
        dst_image_path='image',
        dst_label_filename='train_label.txt',
        annotation_filename='train_full_labels.json',
        nproc=args.n_proc)
    print('Finish')


if __name__ == '__main__':
    main()
