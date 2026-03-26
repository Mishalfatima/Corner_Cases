import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os
import glob
import pickle
import cv2
from PIL import Image
import random
from skimage import io
from skimage.transform import resize
import logging
from datetime import datetime

# --------- Set seeds for reproducibility ----------
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# --------- Setup logging and rest of your script below ---------


# ------------------------ Setup Logging ------------------------
log_file = f'dataset_generation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Dataset creation script started.")

# ------------------------ Constants & Args ------------------------
random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, required=True, help='Job number (0-9)')
parser.add_argument('--random', type=bool, default=False, help='whether random corner or top right')
parser.add_argument('--GT_bboxes', type=str, default='/ceph/mfatima/corner_cases/Inpaint-Anything/val_GT_bboxes.pkl', help='Path to the GT bounding boxes pickle file')
parser.add_argument('--in_path', type=str, default='/ceph/mfatima/corner_cases/Inpainted_ImageNet_GT/test/', help='Path to the inpainted input images directory')
parser.add_argument('--images_path', type=str, default='/ceph/mfatima/corner_cases/ImageNet/val/', help='Path to the images directory')
parser.add_argument('--output_path', type=str, default='/ceph/mfatima/corner_cases/Hard-Spurious-ImageNet-AR/test/', help='Path to the output directory')
args = parser.parse_args()
job_id = args.number

# ------------------------ Load Data ------------------------
with open(args.GT_bboxes, 'rb') as f:
    data = pickle.load(f)
logging.info("Loaded bounding box data.")

in_path = args.in_path
images_path = args.images_path
out_path = args.output_path

img_size = 224
obj_size = [56, 84, 112]
classes = list(data.keys())
resize_counts = {56: 0, 84: 0, 112: 0}

# ------------------------ Class Slicing ------------------------
num_jobs = 10
classes_per_job = len(classes) // num_jobs
start_idx = job_id * classes_per_job
end_idx = len(classes) if job_id == num_jobs - 1 else start_idx + classes_per_job
classes = classes[start_idx:end_idx]
logging.info(f"Job ID: {job_id} | Classes handled: {start_idx} to {end_idx}")

# ------------------------ Helper Functions ------------------------
def resize_short_side_safe_pil(img, target_short_side, max_size=224):
    h, w = img.shape[:2]
    short_side = min(h, w)
    scale = target_short_side / short_side
    new_w, new_h = int(w * scale), int(h * scale)

    flag = False
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_w = int(new_w * scale)
        new_h = int(new_h * scale)
        flag = True

    pil_img = Image.fromarray(img).convert('RGB')
    resized = np.array(pil_img.resize((new_w, new_h))) / 255.
    return resized, flag

def utils(img_size, bg, obj_bbox, obj_bbox_seg=None):
    # Background size
    H, W = img_size, img_size

    # Object bbox shape
    oh, ow, _ = obj_bbox.shape

    # Center placement coordinates
    cx = H // 2
    cy = W // 2

    # Calculate valid placement coordinates for center
    start_x = max(0, cx - oh // 2)
    end_x = min(H, start_x + oh)
    start_y = max(0, cy - ow // 2)
    end_y = min(W, start_y + ow)

    # Corresponding crop in obj_bbox for center placement
    obj_start_x = 0 if cx - oh // 2 >= 0 else abs(cx - oh // 2)
    obj_end_x = obj_start_x + (end_x - start_x)
    obj_start_y = 0 if cy - ow // 2 >= 0 else abs(cy - ow // 2)
    obj_end_y = obj_start_y + (end_y - start_y)

    # Normalize background
    new_image = bg.copy() / 255.

    # Place object bbox centered
    new_image[start_x:end_x, start_y:end_y, :] = obj_bbox[obj_start_x:obj_end_x, obj_start_y:obj_end_y, :]
    final_image_centered = new_image

    H, W = img_size, img_size
    oh, ow, _ = obj_bbox.shape

    # Normalize background
    new_image = bg.copy() / 255.

    if args.random:
        # Randomly choose one of four corners
        corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        corner = random.choice(corners)

    else:
        corner = 'top-right'  # fixed to top-right for non-random case

    if corner == 'top-left':
        start_x, start_y = 0, 0
    elif corner == 'top-right':
        start_x, start_y = 0, max(0, W - ow)
    elif corner == 'bottom-left':
        start_x, start_y = max(0, H - oh), 0
    elif corner == 'bottom-right':
        start_x, start_y = max(0, H - oh), max(0, W - ow)

    end_x = start_x + min(oh, H - start_x)
    end_y = start_y + min(ow, W - start_y)

    obj_start_x = 0
    obj_end_x = end_x - start_x
    obj_start_y = 0
    obj_end_y = end_y - start_y

    # Place object
    new_image[start_x:end_x, start_y:end_y, :] = obj_bbox[obj_start_x:obj_end_x, obj_start_y:obj_end_y, :]
    final_image_random_corner = new_image
    
    '''# Top-right corner placement
    new_image = bg.copy() / 255.

    # For top-right:
    start_x = 0
    end_x = min(H, oh)   # height of object bbox clipped to image height

    # y coords: top-right means aligned to top and right edge
    start_y = max(0, W - ow)  # right edge minus object width (clipped)
    end_y = W  # right edge of image

    # Corresponding crop in obj_bbox for top-right placement
    obj_start_x = 0
    obj_end_x = end_x - start_x
    obj_start_y = 0
    obj_end_y = obj_start_y + (end_y - start_y)

    # Place object bbox at top-right corner
    new_image[start_x:end_x, start_y:end_y, :] = obj_bbox[obj_start_x:obj_end_x, obj_start_y:obj_end_y, :]
    final_image_topright = new_image'''

    return final_image_centered, final_image_random_corner 

def load_img(path, resize=False):
    img = Image.open(path)
    if resize:
        img = img.resize((img_size, img_size))
    if img.mode in ("RGBA", "CMYK"):
        img = img.convert("RGB")
    return np.array(img)

def save_img(im, obj_size, group, j, i):
    g = f'Group_{group}'
    output_dir = os.path.join(out_path, str(obj_size), g, j)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{i.split('/')[-1].split('.')[0]}_{group}_{obj_size}.JPEG"
    out_img_p = os.path.join(output_dir, filename)

    im = im * 255.
    cv2.imwrite(out_img_p, cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))

# ------------------------ Main Loop ------------------------
for c, j in enumerate(classes):
    images = glob.glob(os.path.join(in_path, j + "/*.JPEG"))
    logging.info(f"[{c+1}/{len(classes)}] Processing class: {j} with {len(images)} images.")

    other_id = random.choice([x for x in range(1000) if x != c])
    other_class = classes[other_id % len(classes)]
    other_images = glob.glob(os.path.join(in_path, other_class + "/*.JPEG"))
    if not other_images:
        logging.warning(f"No images in {other_class}, selecting another class.")
        other_class = classes[random.randint(0, len(classes) - 1)]
        other_images = glob.glob(os.path.join(in_path, other_class + "/*.JPEG"))

    res = random.choices(range(len(other_images)), k=len(images)) if len(other_images) < len(images) else random.sample(range(len(other_images)), len(images))

    for cc, i in enumerate(images):
        try:
            Dino_pred_bboxes = data[j]['bbox_GT'][i.split('/')[-1]]
            im_p = os.path.join(images_path, j, i.split('/')[-1])
            img = load_img(im_p)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            h, w, _ = img.shape
            boxes = torch.Tensor(Dino_pred_bboxes) * torch.Tensor([w, h, w, h])
            xyxy = boxes[-1, :].numpy()
            obj_bbox = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
            obj_bbox_seg = None  # unused

            obj_bbox1, resized1 = resize_short_side_safe_pil(obj_bbox, obj_size[0])
            obj_bbox2, resized2 = resize_short_side_safe_pil(obj_bbox, obj_size[1])
            obj_bbox3, resized3 = resize_short_side_safe_pil(obj_bbox, obj_size[2])
            if resized1: resize_counts[56] += 1
            if resized2: resize_counts[84] += 1
            if resized3: resize_counts[112] += 1

            im_inpainted_p = os.path.join(in_path, j, i.split('/')[-1])
            bg = load_img(im_inpainted_p, resize=True)

            # Group 1-2: same class background
            for bbox, size in zip([obj_bbox1, obj_bbox2, obj_bbox3], obj_size):
                center, corner = utils(img_size, bg, bbox, obj_bbox_seg)
                save_img(center, size, 1, j, i)
                save_img(corner, size, 2, j, i)

            # Group 3-4: different class background
            other_im_path = other_images[res[cc]]
            bg = load_img(other_im_path, resize=True)
            for bbox, size in zip([obj_bbox1, obj_bbox2, obj_bbox3], obj_size):
                center, corner = utils(img_size, bg, bbox, obj_bbox_seg)
                save_img(center, size, 3, j, i)
                save_img(corner, size, 4, j, i)

            logging.info(f"Processed image: {i.split('/')[-1]} for class: {j}")

        except Exception as e:
            logging.error(f"Error processing {i}: {e}")

logging.info("Dataset generation complete.")
logging.info(f"Resized due to max-size limit: {resize_counts}")
