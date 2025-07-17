import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torch_scatter import scatter_min
import cv2
import os
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scipy.spatial.transform import Rotation
import csv
from utils.seg_utils import conv2d_matrix, compute_ratios, update, project_to_2d
import torchvision.io as io
import torchvision.transforms.functional as func

def get_global_idx(tsv_file, target_idx, split_type="train"):
    """
    根据给定的 split_type（如 "train" 或 "test"）以及行号（不计 header 和空 id 行），
    返回对应的 id 值作为 global_idx。如果没有找到则返回 None。
    """
    count = 0
    with open(tsv_file, newline='', encoding='utf-8') as f:
        # TSV 文件，分隔符为制表符，第一行为表头
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # 判断 split 类型是否匹配
            if row['split'].strip().lower() != split_type.lower():
                continue
            # 如果 id 字段为空，则跳过
            if not row['id'].strip():
                continue
            if count == target_idx:
                return row['id'].strip()
            count += 1
    return None

def get_filename_by_id(tsv_file, target_id):
    """
    根据目标 id，在 TSV 文件中找到对应的 filename。
    """
    target_id = str(int(float(target_id)))  # 统一转换成整数格式的字符串
    with open(tsv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['id'].strip() and str(int(float(row['id'].strip()))) == target_id:
                return row['filename'].strip()  # 返回对应的图片名称
    return None  # 没找到返回 None

def project_gs_idx(gs_coords, idx, data_path):
    # gs_coords代表高斯点的世界坐标，idx代表对应照片的序号，data_path代表数据存放地址
    cameras_extrinsic_file = os.path.join(data_path, "sparse", "images.bin")
    cameras_intrinsic_file = os.path.join(data_path, "sparse", "cameras.bin")
    cam_extrinsic = read_extrinsics_binary(cameras_extrinsic_file)[idx]
    cam_intrinsic = read_intrinsics_binary(cameras_intrinsic_file)[idx]
    qvec = cam_extrinsic.qvec
    tvec = cam_extrinsic.tvec
    params = cam_intrinsic.params

    # 1. 计算旋转矩阵 R
    R = Rotation.from_quat(qvec[[1, 2, 3, 0]]).as_matrix()  # SciPy 的四元数格式是 xyzw
    # 2. 构造 c2w 矩阵（4x4）
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ tvec
    # 内参矩阵K的计算过程
    fx, fy, cx, cy = params
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    # 如果 gs_coords 是 CUDA 上的 tensor，则转换为 CPU 上的 numpy 数组
    if hasattr(gs_coords, "is_cuda") and gs_coords.is_cuda:
        gs_coords = gs_coords.detach().cpu().numpy()

    cam_pos = c2w[:3, 3]
    pnt_cam = ((gs_coords - cam_pos)[..., None, :] * c2w[:3, :3].transpose()).sum(-1) #点集的相机坐标系坐标
    uv_cam = (pnt_cam / pnt_cam[..., 2:]) @ K.T                                       #点集投影到图像平面上的点坐标
    return uv_cam, pnt_cam, pnt_cam[..., 2:]  # w , h

def project_gs_name(gs_coords, image_name, data_path):
    # gs_coords代表高斯点的世界坐标，idx代表对应照片的序号，data_path代表数据存放地址
    cameras_extrinsic_file = os.path.join(data_path, "sparse", "images.bin")
    cameras_intrinsic_file = os.path.join(data_path, "sparse", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    idx = next((id for id, img in cam_extrinsics.items() if img.name == image_name), None)
    cam_extrinsic = cam_extrinsics[idx]
    cam_intrinsic = cam_intrinsics[idx]
    qvec = cam_extrinsic.qvec
    tvec = cam_extrinsic.tvec
    params = cam_intrinsic.params

    # 1. 计算旋转矩阵 R
    R = Rotation.from_quat(qvec[[1, 2, 3, 0]]).as_matrix()  # SciPy 的四元数格式是 xyzw
    # 2. 构造 c2w 矩阵（4x4）
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ tvec
    # 内参矩阵K的计算过程
    fx, fy, cx, cy = params
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    # 如果 gs_coords 是 CUDA 上的 tensor，则转换为 CPU 上的 numpy 数组
    if hasattr(gs_coords, "is_cuda") and gs_coords.is_cuda:
        gs_coords = gs_coords.detach().cpu().numpy()

    cam_pos = c2w[:3, 3]
    pnt_cam = ((gs_coords - cam_pos)[..., None, :] * c2w[:3, :3].transpose()).sum(-1) #点集的相机坐标系坐标
    uv_cam = (pnt_cam / pnt_cam[..., 2:]) @ K.T                                       #点集投影到图像平面上的点坐标
    return uv_cam, pnt_cam, pnt_cam[..., 2:]  # w , h

#创建一个二维掩码，它可以用来选择或过滤图像中的特定区域，同时考虑深度信息
def mask_pcd_2d(uv, mask, thresh=0.5):
    h, w = mask.shape
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    # uv_rescale = (uv[:, :2] - h_w_half / 2) / h_w_half
    # uv_max = np.max(uv_rescale)
    # uv_min = np.min(uv_rescale)
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    sample_mask = sample_mask > thresh

    return sample_mask

def Spiky_3DGaussians_Cutter(gaussians, scene, input_mask, camera_id):
    with torch.no_grad():
        viewpoint_camera = scene.getTrainCameras()[camera_id]
        # Step 1: 将 3D 高斯点投影到 2D
        # 获取 3D 高斯点的坐标
        xyz = gaussians.get_xyz
        point_image = project_to_2d(viewpoint_camera, xyz)
        # Step 2: 计算哪些点在 mask 内
        h, w = viewpoint_camera.image_height, viewpoint_camera.image_width

        # # 可能需要调整 mask 尺寸以匹配相机尺寸
        # input_mask = io.read_image(os.path.join(args.model_path, "sam_mask.png"))[0]  # 只取单通道 (H, W)
        # 归一化并二值化 (0/1)
        input_mask = torch.from_numpy(input_mask.astype(int))
        input_mask = input_mask.to("cuda")  # 直接移动到 GPU
        if input_mask.shape[0] != h or input_mask.shape[1] != w:
            input_mask = func.resize(input_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
        input_mask = input_mask.long()
        point_image = point_image.long()

        # Step 3: 检查投影点是否在 mask 范围内
        valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
        valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
        valid_mask = valid_x & valid_y
        # 选出 mask 内的点
        point_mask = torch.full((point_image.shape[0],), -1).to("cuda")
        point_mask[valid_mask] = input_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
        # 仅保留前景区域的点
        indices_mask = torch.where(point_mask == 1)[0]

        # Step 4: 计算 2D-3D 的融合权重
        conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
        index_in_all, ratios, dir_vector = compute_ratios(conv2d, point_image, indices_mask, input_mask, h, w)

        # Step 5: 更新高斯点
        decomp_gaussians = update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)

        return decomp_gaussians


if __name__ == '__main__':
    gs_coords = torch.randn(100, 3, device='cuda')
    # project_gs(gs_coords, 1, r"H:\PycharmProject\Gaussian-Wild-main-feature\data\trevi_fountain\dense")