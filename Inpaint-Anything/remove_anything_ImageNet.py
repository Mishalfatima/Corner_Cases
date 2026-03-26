import os
import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os
import glob
import pickle
import pdb
import cv2
from sam_segment import predict_masks_with_sam
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', 
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+',
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--pkl_file_path", type=str, default = './val_GT_bboxes.pkl',
        help="The path to the pkl file containing normalized ground truth bounding boxes.",
    )
    parser.add_argument(
        "--imagenet_path", type=str, default = '/ceph/mfatima/corner_cases/ImageNet/val',
        help="The path to imagenet data",)
    parser.add_argument(
        "--output_path", type=str, default = '/ceph/mfatima/corner_cases/Inpainted_ImageNet_GT/test',
        help="The path to the output directory for inpainted images.",
    )



if __name__ == "__main__":
    """Example usage:
    python remove_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()
    with open(args.pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = args.imagenet_path
    out_path = args.output_path

    classes = data.keys()
    classes = list(classes)
    count = 0

    for j in classes:
    
        image_names = list(data[j]['bbox_GT'].keys())
        image_names_list= image_names

        count+=1
        print(count)
        for i in image_names_list:
            image_path = os.path.join(path, j, i)
            out_img_path = os.path.join(out_path, j, i)

            if os.path.exists(out_img_path):
                continue

            GT_bboxes = data[j]['bbox_GT'][i.split('/')[-1]]

            img = load_img_to_array(image_path)

            if len(img.shape)  == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            h, w, _ = img.shape

            xyxy_GT = torch.Tensor(GT_bboxes) 
            boxes = xyxy_GT * torch.Tensor([w, h, w, h])
            xyxy = boxes.numpy()

            point_labels = [1]
            dilate_kernel_size = 15
            llama_config = './lama/configs/prediction/default.yaml'
            lama_ckpt =  './pretrained_models/big-lama'

            for ii in range(xyxy.shape[0]):

                masks, scores, logits = predict_masks_with_sam(
                    img,
                    None,
                    point_labels,
                    model_type='vit_h',
                    ckpt_p='./pretrained_models/sam_vit_h_4b8939-001.pth',
                    device=device,
                    boxes = np.expand_dims(xyxy[ii,:], 0))
                                        
                masks = masks.astype(np.uint8) * 255

                # dilate mask to avoid unmasked edge effect

                if dilate_kernel_size is not None:
                    masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

                mask = masks[np.argmax(scores)]

                img_inpainted = inpaint_img_with_lama(
                    img, mask, llama_config, lama_ckpt, device=device)
                
                img = img_inpainted
            

            if not os.path.exists(os.path.join(out_path, j)):
                os.makedirs(os.path.join(out_path, j))

            img_inpainted_p = os.path.join(out_path, j, i)

            import pdb; pdb.set_trace()
            save_array_to_img(img_inpainted, img_inpainted_p)
