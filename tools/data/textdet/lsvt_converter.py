# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from functools import partial

import cv2
import mmcv
import numpy as np
from shapely.geometry import Polygon

from mmocr.utils import convert_annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of LSVT')
    parser.add_argument(
        'root_path', type=str, help='The path to the directory of LSVT')
    parser.add_argument(
        'n_proc', default=1, type=int, help='Number of processes to run')
    args = parser.parse_args()
    return args


def process_info(raw_ann_pair, img_root):
    img_name, img_anns = raw_ann_pair
    img_path = osp.join(img_root, f'{img_name}.jpg')
    img = mmcv.imread(img_path, 'unchanged')

    anno_info = []

    for ann in img_anns:
        segmentation = []
        for pt in ann['points']:
            segmentation.append(max(pt[0], 0))
            segmentation.append(max(pt[1], 0))
        bbox = cv2.boundingRect(np.array(ann['points']))
        anno = dict(
            iscrowd=ann['illegibility'],
            category_id=1,
            bbox=bbox,
            area=Polygon(ann['points']).area,
            segmentation=[segmentation])
        anno_info.append(anno)

    img_info = dict(
        file_name=osp.join(f'{osp.basename()}/{img_name}.jpg'),
        height=img.shape[0],
        width=img.shape[1],
        anno_info=anno_info,
        segm_file='',
    )
    return img_info


def collect_lsvt_info(root_path, nproc):

    annotation_path = osp.join(root_path, 'train_full_labels.json')
    img_root = osp.join(root_path, 'train_full_images')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} does not exist, please check and try again.')

    annotation = mmcv.load(annotation_path)
    process_info_with_img_root = partial(process_info, img_root=img_root)
    img_infos = mmcv.track_parallel_progress(
        process_info_with_img_root, annotation.items(), nproc=nproc)
    return img_infos


def main():
    args = parse_args()
    print('Processing training set...')
    training_infos = collect_lsvt_info(args.root_path, args.n_proc)
    convert_annotations(training_infos,
                        osp.join(args.root_path, 'instances_training.json'))
    print('Finish')


if __name__ == '__main__':
    main()
