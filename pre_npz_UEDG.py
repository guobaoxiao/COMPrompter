# %% import packages
import numpy as np
import os
from glob import glob
import pandas as pd
import cv2
join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from PIL import Image

# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")

# add arguments to the parser
parser.add_argument(
    "-i",
    "--img_path",
    type=str,
    default="/data/Test/CHAMELEON/im",
    help="path to the images",
)
parser.add_argument(
    "-gt",
    "--gt_path",
    type=str,
    default="/data/Test/CHAMELEON/gt",
    help="path to the ground truth (gt)",
)

parser.add_argument(
    "-gt2",
    "--gtBSA_path",
    type=str,
    default="data/edge",
    help="path to the ground truth (gt)",
)


parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="path to the csv file",
)

parser.add_argument(
    "-o",
    "--npz_path",
    type=str,
    default="data/demo2D",
    help="path to save the npz files",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="CHAMELEON",
    help="dataset name; used to name the final npz file, e.g., demo2d.npz",
)
parser.add_argument("--image_size", type=int, default=256, help="image size")
parser.add_argument(
    "--img_name_suffix", type=str, default=".jpg", help="image name suffix"
)
parser.add_argument("--label_id", type=int, default=255, help="label id")
# parser.add_argument("--label_id_BSA", type=int, default=200, help="label id")
parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="work_dir/SAM/sam_vit_b_01ec64.pth",
    help="checkpoint",
)
parser.add_argument("--device", type=str, default="cuda:1", help="device")
parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()

# convert 2d grey or rgb images to npz file
imgs = []
gts = []
number = []
boundary = []
img_embeddings = []
global num_of_processed_imgs
num_of_processed_imgs = 0


sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(
    args.device
)
save_path = args.npz_path + "_" + args.model_type
os.makedirs(save_path, exist_ok=True)

def find_boundary(img, mask, gt, path, name):
    # Convert mask to 0-255 range
    mask = mask * 255

    # Detect edges using Canny edge detector
    edges = cv2.Canny(img, 0.2, 0.6)

    # Create necessary directories
    os.makedirs(os.path.join(path, 'canny'), exist_ok=True)
    cv2.imwrite(os.path.join(path, 'canny', name), edges)

    # Load the saved edge image and convert to binary
    edges_2 = Image.open(os.path.join(path, 'canny', name)).convert('1')

    # Find bounding box from ground truth
    y_indices, x_indices = np.where(gt > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = np.array([x_min, y_min, x_max, y_max])

    # Remove edges outside the bounding box
    edges_2_np = np.array(edges_2)
    edges_2_np[:y_min, :] = 0
    edges_2_np[y_max:, :] = 0
    edges_2_np[:, :x_min] = 0
    edges_2_np[:, x_max:] = 0
    edges_2 = Image.fromarray(edges_2_np)

    # Multiply mask and edges to get the final boundary
    mask_np = np.array(mask)
    boundary_grads = Image.fromarray((mask_np * edges_2_np).astype(np.uint8))

    # Save the final boundary image
    os.makedirs(os.path.join(path, 'boundary_grads'), exist_ok=True)
    boundary_grads.save(os.path.join(path, 'boundary_grads', name))

    return boundary_grads

def process(gt_name: str, image_name: str, num_of_processed_imgs:int):
    if image_name == None:
        image_name = gt_name.split(".")[0] + args.img_name_suffix  # Find the name of images based on the name of GT
    gt_data = io.imread(join(args.gt_path, gt_name))
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    gt_data = transform.resize(
        gt_data == args.label_id,
        (args.image_size, args.image_size),
        order=0,
        preserve_range=True,
        mode="constant",
    )
    # convert to uint8
    gt_data = np.uint8(gt_data)

    if np.sum(gt_data) > 5:  # exclude tiny objects(Polyps may be small in shape)
        """Optional binary thresholding can be added"""
        assert (
            np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2
        ), "ground truth should be binary"

        image_data = io.imread(join(args.img_path, image_name))
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(
            image_data, 99.5
        )
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0

        image_data_pre = transform.resize(
            image_data_pre,
            (args.image_size, args.image_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
        image_data_pre = np.uint8(image_data_pre)

        imgs.append(image_data_pre)
        number.append(image_name)

        print("the number of images: " + str(len(imgs)) + " and the name of image: " + str(image_name))
        num_of_processed_imgs = num_of_processed_imgs + 1
        assert np.sum(gt_data) > 5, "ground truth should have more than 50 pixels"

        gts.append(gt_data)
# --------------------------------------------------------------
        # 读取图像
        gtBSA_data = io.imread(join(args.gtBSA_path, gt_name))

        # 如果图像是彩色的，转换为灰度图像
        if len(gtBSA_data.shape) == 3:
            gtBSA_data = gtBSA_data[:, :, 0]

        # 确保图像是二维的
        assert len(gtBSA_data.shape) == 2, "ground truth should be 2D"

        # 计算原始图像的最大值和最小值
        print(f"Original max value of gtBSA_data: {np.max(gtBSA_data)}")
        print(f"Original min value of gtBSA_data: {np.min(gtBSA_data)}")

        # 计算直方图
        hist, bins = np.histogram(gtBSA_data.flatten(), bins=256, range=[0, 256])

        # 找到占比最大的像素值
        most_frequent_value = bins[np.argmax(hist)]

        print(f"Most frequent pixel value: {most_frequent_value}")

        # 将最大频率的像素值作为 threshold_min
        threshold_min = most_frequent_value + 5
        # threshold_min = 0
        threshold_max = 255

        # 应用阈值处理：大于 threshold_min 且小于 threshold_max 的像素值设为 1，其余设为 0
        binary_mask = np.where((gtBSA_data > threshold_min) & (gtBSA_data <= threshold_max), 1, 0)
        # convert to uint8
        binary_mask = np.uint8(binary_mask)

        # Output the max and min values of gtBSA_data
        print(f"Max value of resized gtBSA_data: {np.max(binary_mask)}")
        print(f"Min value of resized gtBSA_data: {np.min(binary_mask)}")

        gtBSA_data = cv2.resize(binary_mask, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

        # convert to uint8
        gtBSA_data = np.uint8(gtBSA_data)

        # Output the max and min values of gtBSA_data
        print(f"Max value of resized gtBSA_data: {np.max(gtBSA_data)}")
        print(f"Min value of resized gtBSA_data: {np.min(gtBSA_data)}")

        if np.sum(gtBSA_data) > 0:  # exclude tiny objects(Polyps may be small in shape)
            """Optional binary thresholding can be added"""
            assert (
                    np.max(gtBSA_data) == 1 and np.unique(gtBSA_data).shape[0] == 2
            ), "ground truth should be binary"
        # -----------------------------------------------------------------------------
        boundary.append(find_boundary(image_data_pre, gtBSA_data, gt_data, save_path+'/'+args.data_name, gt_name))

        # resize image to 3*1024*1024
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(
            args.device
        )
        input_image = sam_model.preprocess(
            resize_img_tensor[None, :, :, :]
        )  # (1, 3, 1024, 1024)
        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        # pre-compute the image embedding
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embeddings.append(embedding.cpu().numpy()[0])

    return num_of_processed_imgs


if args.csv != None:
    # if data is presented in csv format
    # columns must be named image_filename and mask_filename respectively
    try:
        os.path.exists(args.csv)
    except FileNotFoundError as e:
        print(f"File {args.csv} not found!!")

    df = pd.read_csv(args.csv)
    bar = tqdm(df.iterrows(), total=len(df))
    for idx, row in bar:
        process(row.mask_filename, row.image_filename)

else:
    names = sorted(os.listdir(args.gt_path))
    # print the number of images found in the ground truth folder
    print("image number:", len(names))
    for gt_name in tqdm(names):
        num_of_processed_imgs = process(gt_name, None, num_of_processed_imgs)
    print("the number of processed images: " + str(num_of_processed_imgs))


# stack the list to array
print("Num. of images:", len(imgs))
if len(imgs) > 1:
    imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
    gts = np.stack(gts, axis=0)  # (n, 256, 256)
    img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
    boundary = np.stack(boundary, axis=0)  # (n, 256, 256)
    np.savez_compressed(
        join(save_path, args.data_name + ".npz"),
        imgs=imgs,
        gts=gts,
        number=number,
        img_embeddings=img_embeddings,
        boundary=boundary
    )
    # save an example image for sanity check
    idx = np.random.randint(imgs.shape[0])
    img_idx = imgs[idx, :, :, :]
    gt_idx = gts[idx, :, :]
    bd = segmentation.find_boundaries(gt_idx, mode="inner")
    img_idx[bd, :] = [255, 0, 0]
    io.imsave(save_path + ".png", img_idx, check_contrast=False)
else:
    print(
        "Do not find image and ground-truth pairs. Please check your dataset and argument settings"
    )