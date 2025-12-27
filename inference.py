import os

import glob

import torch

import numpy as np

import matplotlib.pyplot as plt

import nibabel as nib

from monai.networks.nets import vista3d132

from monai.inferers import sliding_window_inference

from monai.transforms import (

    Compose, LoadImage, EnsureChannelFirst, Orientation,

    Spacing, ScaleIntensityRange, Activations, AsDiscrete

)

 

# --- CONFIGURATION ---

PROJECT_ROOT = "/home/dem7clj/repos/mirpr"

DATA_ROOT = "/shares/CC_v_Val_FV_Gen3_all/VIDT_DL/data/cnn_training/projects/smart_data_selection/town_signs/preprocessed_output_sota_luna16/"

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "runs/vista3d_lidc_finetune/best_model.pth")

 

# VISTA Parameters

TARGET_CLASS = 23  # Lung Tumor

ROI_SIZE = (192, 192, 192)

TARGET_SPACING = (0.7, 0.7, 0.7)

INTENSITY_RANGE = (-1000, 1000)

 

import json

 

def find_small_nodule_sample(data_root):

    """

    Scans ONLY the validation set from dataset.json for a 'Small' nodule.

    """

    json_path = os.path.join(data_root, "dataset.json")

   

    # 1. Load the Split List

    with open(json_path, 'r') as f:

        data = json.load(f)

       

    val_files = data["validation"] # <--- SAFETY LOCK

    print(f"Scanning {len(val_files)} VALIDATION samples for a small nodule...")

 

    # 2. Iterate only through validation files

    for entry in val_files:

        lbl_path = entry["label"]

       

        # Ensure path is absolute

        if not os.path.exists(lbl_path):

            lbl_path = os.path.join(data_root, lbl_path)

           

        # Check volume

        lbl_vol = nib.load(lbl_path).get_fdata()

        lbl_obj = nib.load(lbl_path)

        print(f"Spacing on disk: {lbl_obj.header.get_zooms()}") # Check if it prints (0.7, 0.7, 0.7) or (1.5, 1.5, 1.5)

        voxel_count = np.sum(lbl_vol)

 

        # Criteria: Small (10 - 300 voxels)

        if 10 < voxel_count < 100:

            img_path = entry["image"]

            if not os.path.exists(img_path):

                img_path = os.path.join(data_root, img_path)

               

            img_filename = os.path.basename(img_path)

            print(f"Found VALIDATION Candidate: {img_filename}")

            print(f" -> Nodule Volume: {voxel_count:.0f} voxels")

            return img_path, lbl_path

 

    print("Warning: No 'Small' nodule found in validation set. Returning first positive validation sample.")

   

    # Fallback: Return first positive validation sample

    for entry in val_files:

        lbl_path = entry["label"]

        if not os.path.exists(lbl_path): lbl_path = os.path.join(data_root, lbl_path)

       

        if np.sum(nib.load(lbl_path).get_fdata()) > 0:

            img_path = entry["image"]

            if not os.path.exists(img_path): img_path = os.path.join(data_root, img_path)

            return img_path, lbl_path

 

    raise FileNotFoundError("No positive samples found in validation set!")

 

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   

    # 1. Get Small Nodule Data

    img_path, lbl_path = find_small_nodule_sample(DATA_ROOT)

    print(f"Using Image: {img_path}")

 

    # 2. Define Transforms

    preprocess = Compose([

        LoadImage(image_only=True),

        EnsureChannelFirst(),

        Orientation(axcodes="RAS"),

        Spacing(pixdim=TARGET_SPACING, mode="bilinear"),

        ScaleIntensityRange(

            a_min=INTENSITY_RANGE[0], a_max=INTENSITY_RANGE[1],

            b_min=0.0, b_max=1.0, clip=True

        ),

    ])

   

    preprocess_lbl = Compose([

        LoadImage(image_only=True),

        EnsureChannelFirst(),

        Orientation(axcodes="RAS"),

        Spacing(pixdim=TARGET_SPACING, mode="nearest"),

    ])

 

    print("Preprocessing...")

    img_tensor = preprocess(img_path).unsqueeze(0).to(device)

    lbl_tensor = preprocess_lbl(lbl_path).unsqueeze(0).to(device)

 

    # 3. Load Model

    print(f"Loading weights...")

    model = vista3d132(encoder_embed_dim=48, in_channels=1).to(device)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))

    model.eval()

 

    # 4. Inference

    print("Running Inference...")

    prompt_class = torch.tensor([TARGET_CLASS], device=device)

   

    with torch.no_grad():

        with torch.amp.autocast('cuda'):

            logits = sliding_window_inference(

                inputs=img_tensor,

                roi_size=ROI_SIZE,

                sw_batch_size=4,

                predictor=model,

                overlap=0.5,

                transpose=True,

                class_vector=prompt_class

            )

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()

 

    # 5. Visualization

    img_np = img_tensor[0, 0].cpu().numpy()

    gt_np = lbl_tensor[0, 0].cpu().numpy()

    pred_np = preds[0, 0].cpu().numpy()

 

    # Find the specific slice containing the center of the nodule

    non_zeros = np.argwhere(gt_np > 0)

    if len(non_zeros) > 0:

        z_center = int(np.median(non_zeros[:, 2]))

        y_center = int(np.median(non_zeros[:, 1]))

        x_center = int(np.median(non_zeros[:, 0]))

        print(f"Visualizing Nodule Center at: x={x_center}, y={y_center}, z={z_center}")

       

        # Crop zoom area (64x64 window around nodule)

        crop_size = 32

        y_min = max(0, y_center - crop_size)

        y_max = min(img_np.shape[1], y_center + crop_size)

        x_min = max(0, x_center - crop_size)

        x_max = min(img_np.shape[0], x_center + crop_size)

    else:

        z_center = img_np.shape[2] // 2

        y_min, y_max, x_min, x_max = 0, img_np.shape[1], 0, img_np.shape[0]

 

    # 6. Plotting (Zoomed In)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

   

    # Helper to crop and display

    def show_slice(ax, vol, overlay=None, title=""):

        # Crop

        crop = vol[x_min:x_max, y_min:y_max, z_center]

        ax.imshow(crop, cmap="gray")

        if overlay is not None:

            overlay_crop = overlay[x_min:x_max, y_min:y_max, z_center]

            ax.imshow(overlay_crop, cmap="jet", alpha=0.5)

        ax.set_title(title)

        ax.axis('off')

 

    show_slice(axes[0], img_np, title=f"Input CT (Zoomed)\nZ={z_center}")

    show_slice(axes[1], img_np, gt_np, title="Ground Truth")

    show_slice(axes[2], img_np, pred_np, title="VISTA Prediction")

 

    plt.tight_layout()

    output_png = "vista_small_nodule_our_checkpoint_6.png"

    plt.savefig(output_png)

    print(f"\nSaved zoomed visualization to: {os.path.abspath(output_png)}")

 

if __name__ == "__main__":

    main()