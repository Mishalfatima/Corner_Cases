import os
import shutil

# ------------------------ Paths ------------------------
src_root  = '/ceph/mfatima/corner_cases/dataset/Hard-Spurious-ImageNet/train'
dst_val   = '/ceph/mfatima/corner_cases/dataset/Hard-Spurious-ImageNet/val'

train_list = 'train.txt'
val_list   = 'val.txt'

# ------------------------ Move files ------------------------
def move_files(file_list, dst_root):
    moved, missing = 0, 0
    with open(file_list) as f:
        for line in f:
            rel_path = line.strip()  # /84/Group_3/n02113799/n02113799_1144_3_84.JPEG
            if rel_path.startswith('.'):
                rel_path = rel_path[1:]  # Remove leading . if present

            import pdb; pdb.set_trace()
            src = src_root + rel_path
            dst = dst_root + rel_path
            if not os.path.exists(src):
                print(f"Missing: {src}")
                missing += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            import pdb; pdb.set_trace()
            shutil.move(src, dst)
            moved += 1
    print(f"Moved: {moved} | Missing: {missing}")

move_files(val_list, dst_val)