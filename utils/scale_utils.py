import open3d as o3d
import numpy as np
import struct
import os
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch
import sys
from plyfile import PlyData, PlyElement
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from arguments import ModelParams, PipelineParams, get_combined_args,args_init
from gaussian_renderer import GaussianModel
from utils.general_utils import *
from scene import Scene
from typing import Any, Dict, Generator, ItemsView, List, Tuple
from itertools import product


def calculate_distance(path):
    cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
    # ply_bin_file = os.path.join(path, "sparse/points3D.bin")
    # xyz, rgb, _ = read_points3D_binary(ply_bin_file)

    ply_file = os.path.join(path, "sparse/points3D.ply")
    plydata = PlyData.read(ply_file)
    vertices = plydata['vertex']
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    camera_centers = torch.empty(0, 3).cuda()
    camera_directions = torch.empty(0, 3).cuda()
    print(len(cam_extrinsics))

    offset = 0
    distance_info = {}  # 用于存储照片名称和距离的字典
    for idx, _ in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{} \n".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()
        key = idx + 1
        # if key >= 756:
        #     key += 1
        #     if key >= 1077:
        #         key += 1
        # extr = cam_extrinsics[key]
        # intr = cam_intrinsics[extr.id]
        try:
            cam_extrinsics[key + offset]
        except KeyError:
            offset += 1

        extr = cam_extrinsics[key + offset]
        intr = cam_intrinsics[extr.id]
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        camera_center = world_view_transform.inverse()[3, :3]
        # 将相机中心添加到 camera_centers 列表
        # camera_centers = torch.cat((camera_centers, camera_center.unsqueeze(0)), dim=0)

        camera_direction = torch.tensor(R[:, 2]).cuda()

        # # 方法一： 平均欧式距离（但是没有负数，有点不合理）--------------------------------------------------------
        # if not isinstance(xyz, torch.Tensor):
        #     xyz = torch.from_numpy(xyz).to(camera_direction.dtype).to("cuda")
        # # 计算当前相机中心与所有点云之间的欧几里得距离
        # distances = torch.norm(xyz - camera_center, dim=1)  # L2 范数，形状为 (N,)
        #
        # # 计算平均距离
        # average_distance = torch.mean(distances)
        #
        # if key > 715:
        #     key -= 1
        # # # 存储计算的平均距离
        # # average_distances[key] = average_distance
        # print(f"第{key}张图像的平均距离是：{average_distance}")
        # # ----------------------------------------------------------------------------------------

        # 方法二：求所有点在光轴上的投影值计算距离-----------------------------------------------------------
        # 计算相机中心到每个点云的向量
        if not isinstance(xyz, torch.Tensor):
            xyz = torch.from_numpy(xyz).to(camera_direction.dtype).to("cuda")
        vectors = xyz - camera_center  # 形状为 (N, 3)
        # 计算每个向量在光轴上的投影长度（点积）
        projection_lengths = torch.matmul(vectors, camera_direction)  # 点积，得到形状为 (N,)
        # print(projection_lengths.max(), projection_lengths.min(), f"点积为负数的个数{(projection_lengths < 0).sum().item()}")
        # if key >= 756:
        #     key -= 1
        #     if key >= 1077:
        #         key -= 1
        # 统一焦距，消除焦距对距离估计的影响------------------------------------------------------------------------------
        focal_length_x = 0
        focal_length_y = 0
        height = intr.height
        width = intr.width
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]

        standard_focal_length = 50  # 设置标准焦距为50mm
        standard_sensor_width = 36  # 统一传感器宽度（全画幅，单位：毫米）
        standard_sensor_height = 24  # 统一传感器高度（全画幅，单位：毫米）
        camera_focal_length_x = focal_length_x * (standard_sensor_width / width)
        camera_focal_length_y = focal_length_y * (standard_sensor_height / height)
        camera_focal_length = (camera_focal_length_x + camera_focal_length_y) / 2  # 归一化焦距系数
        # --------------------------------------------------------------------------------------------------------
        distance = projection_lengths.mean().item()
        distance = distance * (standard_focal_length / camera_focal_length)

        # 将照片名称和距离信息存入字典
        distance_info[extr.name] = distance

        # print(f"第{key}张图像:" + extr.name + f"的距离是：{distance}")
        # ----------------------------------------------------------------------------------------------

        # #方法三：求整体点云的质心作为点云中心点，之后计算相机距离点云质心的距离-----------------------------
        # # 计算点云质心
        # if not isinstance(xyz, torch.Tensor):
        #     xyz = torch.from_numpy(xyz).to(camera_direction.dtype).to("cuda")
        # point_cloud_center = torch.mean(xyz, dim=0)
        # # 计算相机中心与点云质心之间的欧几里得距离
        # projection_lengths = torch.norm(camera_center - point_cloud_center)  # L2 范数即为欧几里得距离
        # if key > 715:
        #     key -= 1
        # print(f"第{key}张图像的距离是：{projection_lengths.mean().item()}")
        # # -------------------------------------------------------------------------------------

        camera_directions = torch.cat((camera_directions, camera_direction.unsqueeze(0)), dim=0)
        camera_centers = torch.cat((camera_centers, camera_center.unsqueeze(0)), dim=0)

    # # 打印所有照片的距离信息
    # print("\n所有照片的距离信息：")
    # for name, dist in distance_info.items():
    #     print(f"{name}: {dist:.2f}")

    # 初始化结果数组
    train_distances = []
    # tsv_path = os.path.join(os.path.dirname(path), path.split("\\")[-2].split("_")[0] + ".tsv")
    base_name = os.path.basename(os.path.dirname(path))
    prefix = base_name.split("_")[0]
    tsv_path = os.path.join(os.path.dirname(path), prefix + ".tsv")
    with open(tsv_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')  # 以 '\t' 作为分隔符读取 TSV
        for row in reader:
            filename = row['filename']
            split = row['split']
            id_ = row['id']

            # 检查 id 是否为空，跳过无效数据
            if not id_ or filename not in distance_info:
                continue
            # 如果 split 为 'train'，将对应的距离加入数组
            if split == 'train' and filename in distance_info:
                train_distances.append(distance_info[filename])

    # 输出结果
    print("Train distances shape:", len(train_distances))
    return np.array(train_distances)


def calculate_scale(path, target_min, target_max):
    distances = calculate_distance(path)
    # Step 1: 归一化到 [0, 1] 范围
    min_val, max_val = distances.min(), distances.max()
    normalized_distances = (distances - min_val) / (max_val - min_val)
    # 动态调整points_per_side
    # 线性映射并取整
    scale_val = [
        round(target_min + (x - 0) * (target_max - target_min) / (1 - 0))
        for x in normalized_distances
    ]
    return scale_val


def classify_distances(distances, num_levels=5):
    """
    将距离数组分为指定的等级。

    参数：
    distances (list or np.ndarray): 一维距离数组。
    num_levels (int): 分组等级数，默认为4。

    返回：
    - levels: 每个距离对应的等级（从1到num_levels）。
    - boundaries: 分组的边界值（归一化后的值）。
    """
    distances = np.array(distances)

    # Step 1: 归一化到 [0, 1] 范围
    min_val, max_val = distances.min(), distances.max()
    normalized_distances = (distances - min_val) / (max_val - min_val)

    # Step 2: 根据分位数进行分组
    boundaries = np.linspace(0, 1, num_levels + 1)  # 等分区间
    levels = np.digitize(normalized_distances, boundaries, right=False)  # 获取每个距离的等级

    # 确保最高等级为 num_levels
    levels[levels > num_levels] = num_levels

    return levels, boundaries

#=====================================================用于生成天空遮罩sky_mask=========================================
from assets.DeepLabV3Plus_Pytorch_master import network
from torchvision import transforms as T
from PIL import Image
import cv2
from assets.DeepLabV3Plus_Pytorch_master import network

# 初始化sky分割模型（只加载一次）
def load_model():
    # 设置模型存放路径
    torch.hub.set_dir(r"./assets")
    # 配置参数
    class Config:
        model = 'deeplabv3plus_mobilenet'  # 使用 mobilenet 作为骨干网络
        separable_conv = False
        output_stride = 16
        dataset = 'cityscapes'
        ckpt = r"./assets/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
        gpu_id = '0'  # 指定 GPU
    opts = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = network.modeling.__dict__[opts.model](num_classes=19, output_stride=opts.output_stride)
    checkpoint = torch.load(opts.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint["model_state"])
    model = torch.nn.DataParallel(model).to(device).eval()

    return model

# 预处理函数
def preprocess_image(image_path):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
    return img_tensor, original_size

def gen_sky_mask(image_path, MODEL, confidence_threshold=0.6):
    img_tensor, original_size = preprocess_image(image_path)
    with torch.no_grad():
        logits = MODEL(img_tensor)  # (1, 19, H, W) -> 每个类别的分数
        probs = torch.softmax(logits, dim=1)  # 归一化得到概率
        pred_mask = probs.argmax(1).cpu().numpy()[0]  # 直接分类的结果（类别ID）
        sky_confidence = probs[0, 10].cpu().numpy()  # 获取类别 10（天空）的置信度

    # 调整 mask（只把置信度高的天空去掉）
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    sky_confidence_resized = cv2.resize(sky_confidence, original_size, interpolation=cv2.INTER_LINEAR)

    # 生成 mask：置信度大于阈值时设为 0（去除天空），否则设为 1（保留）
    mask = np.where((pred_mask_resized == 10) & (sky_confidence_resized > confidence_threshold), 0, 1).astype(np.uint8)
    # #可视化
    # plt.figure(figsize=(6, 6))
    # plt.imshow(mask, cmap="gray")  # 使用灰度 colormap
    # plt.title("Sky Mask (Binary)")
    # plt.axis("off")
    # plt.show()

    return mask
#===================================================================================================================

#=======================================生成深度图====================================================================
def init_scene(args, dataset: ModelParams, iteration: int):
    """ 初始化 GaussianModel 和 Scene，仅执行一次 """
    gaussians = GaussianModel(dataset.sh_degree, args)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return scene, gaussians

def gen_depth_map(scene, gaussians, camera_id):
    with torch.no_grad():
        viewpoint_camera = scene.getTrainCameras()[camera_id]
        xyz = gaussians.get_xyz
        # 获取相机参数
        w2c_matrix = viewpoint_camera.world_view_transform  # 世界到相机坐标系
        intrinsic_matrix = viewpoint_camera.intrinsic_matrix  # 内参矩阵 (3x3)

        # 1. 将 3D 点转换到相机坐标系
        xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=1)  # (N, 4)
        xyz_camera = (xyz_homo @ w2c_matrix)[:, :3]  # (N, 3) -> 只保留 (x, y, z)

        # 2. 透视投影到图像平面
        uv_homo = (xyz_camera @ intrinsic_matrix.T)  # (N, 3)
        uv = uv_homo[:, :2] / (uv_homo[:, 2:3] + 1e-6)  # 归一化除以深度 (z)

        # print("uv 最小值:", uv.min(dim=0).values)
        # print("uv 最大值:", uv.max(dim=0).values)

        # 3. 归一化到图像坐标
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
        uv = torch.round(uv).long()  # 取整

        # 4. 生成深度图
        depth_map = torch.full((H, W), float('inf'), device=xyz.device)  # 初始化深度图（无穷大）

        # 5. 遍历所有点，把最近的深度值填充到图像上
        valid_mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)  # 过滤超出范围的点
        uv_valid = uv[valid_mask]
        depth_valid = xyz_camera[:, 2][valid_mask]  # 获取对应的 z 值（深度）
        depth_map[uv_valid[:, 1], uv_valid[:, 0]] = torch.minimum(depth_map[uv_valid[:, 1], uv_valid[:, 0]],
                                                                  depth_valid)
        print(depth_map[depth_map != float('inf')].min(), depth_map[depth_map != float('inf')].max())
        # 6. 归一化深度值，填充无效区域
        depth_map[depth_map == float('inf')] = depth_map[depth_map != float('inf')].min()  # 无效区域合理化
        norm_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)  # 归一化 0~1

        norm_depth_map = denoise(norm_depth_map)
        return depth_map, norm_depth_map

def denoise(depth_map, lower_threshold=0.01, upper_threshold=0.99):
    # 计算深度图的分位数
    lower_bound = torch.quantile(depth_map, lower_threshold)  # 1%分位
    upper_bound = torch.quantile(depth_map, upper_threshold)  # 99%分位

    print(f"Lower bound: {lower_bound.item()}, Upper bound: {upper_bound.item()}")

    # 对深度图进行剪裁：将低于 lower_bound 的设为 lower_bound，高于 upper_bound 的设为 upper_bound
    filtered_depth_map = depth_map.clone()
    filtered_depth_map[filtered_depth_map < lower_bound] = lower_bound
    filtered_depth_map[filtered_depth_map > upper_bound] = upper_bound

    # 再归一化到 [0, 1]
    depth_map = (filtered_depth_map - lower_bound) / (upper_bound - lower_bound + 1e-6)
    return depth_map

#==========================================生成分割块===============================================================
def auto_divide_layers(depth_map: torch.Tensor, target_blocks=6):
    """根据输入图像尺寸自适应计算三层结构"""
    h, w = depth_map.shape

    # **第三层（细分层，固定16×16）**
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    depth_map_padded = torch.nn.functional.pad(depth_map, (0, pad_w, 0, pad_h), mode='constant', value=0)
    H_new, W_new = depth_map_padded.shape
    third_layer = depth_map_padded.reshape(H_new // 16, 16, W_new // 16, 16).swapaxes(1, 2)

    # **计算第二层分割数量**
    aspect_ratio = W_new / H_new
    if aspect_ratio >= 1:
        sec_w = target_blocks
        sec_h = max(1, round(target_blocks / aspect_ratio))
    else:
        sec_h = target_blocks
        sec_w = max(1, round(target_blocks * aspect_ratio))
    second_layer = depth_map_padded.reshape(sec_h, H_new // sec_h, sec_w, W_new // sec_w).swapaxes(1, 2)

    # **第一层分割数量（比第二层少一半）**
    first_h = max(1, sec_h // 2)
    first_w = max(1, sec_w // 2)
    first_layer = depth_map_padded.reshape(first_h, H_new // first_h, first_w, W_new // first_w).swapaxes(1, 2)

    return first_layer, second_layer, third_layer, depth_map_padded


def compute_cell_depth_values(layer):
    """计算每个单元格的深度总和"""
    return torch.sum(layer, dim=(-1, -2))  # 在最后两个维度 (H_cell, W_cell) 求和

def allocate_sampling_points(depth_values, total_samples):
    """按深度比例分配采样点"""
    depth_sum = torch.sum(depth_values)  # 所有单元格的深度总和
    if depth_sum == 0:
        return torch.zeros_like(depth_values, dtype=torch.int32)  # 避免除零错误

    allocation_ratio = depth_values / depth_sum  # 计算每个单元格的深度比例
    allocated_points = torch.round(allocation_ratio * total_samples).int()  # 四舍五入分配点数

    # **修正误差**
    diff = total_samples - allocated_points.sum()
    if diff > 0:
        idx = torch.argmax(depth_values)  # 选最大深度的像素补足
        allocated_points.view(-1)[idx] += diff

    return allocated_points


def distribute_samples(previous_layer_samples, current_layer_depths, scale_factor=3):
    """根据上一层的采样点数量，分配给当前层"""
    current_layer_samples = torch.zeros_like(current_layer_depths, dtype=torch.int32)
    prev_h, prev_w = previous_layer_samples.shape
    cur_h, cur_w = current_layer_samples.shape
    scale_factor = int(cur_w / prev_w)
    for i in range(prev_h):
        for j in range(prev_w):
            if previous_layer_samples[i, j] > 0:
                h_start, h_end = i * scale_factor, (i + 1) * scale_factor
                w_start, w_end = j * scale_factor, (j + 1) * scale_factor
                current_layer_samples[h_start:h_end, w_start:w_end] += allocate_sampling_points(
                    current_layer_depths[h_start:h_end, w_start:w_end], previous_layer_samples[i, j]
                )
    return current_layer_samples


def generate_point_grids(last_layer_samples, image_w, image_h):
    """
    根据 third_layer_samples 生成采样点，并随机分布到 16×16 的像素格内（整数坐标）
    """
    h, w = last_layer_samples.shape  # 获取层的尺寸
    cell_size = 16  # 每个单元格 16x16

    point_grids = []  # 存储所有采样点
    for i in range(h):
        for j in range(w):
            num_samples = last_layer_samples[i, j].item()  # 获取当前单元格需要的采样点数

            if num_samples > 0:
                # 生成 num_samples 个随机整数 (x, y) 坐标, 在 [0, 15] 范围
                rand_x = np.random.randint(0, cell_size, num_samples)
                rand_y = np.random.randint(0, cell_size, num_samples)

                # 转换到全局坐标系 (整数)
                global_x = j * cell_size + rand_x  # 列方向
                global_y = i * cell_size + rand_y  # 行方向

                # 组合点 (x, y)
                points = np.stack([global_x, global_y], axis=-1)
                point_grids.append(points)

    if point_grids:
        depth_sample_points = np.concatenate(point_grids, axis=0).astype(np.float32)
        # 归一化到 [0,1]
        depth_sample_points[:, 0] /= image_w  # x 归一化
        depth_sample_points[:, 1] /= image_h  # y 归一化
        return depth_sample_points
    else:
        return np.empty((0, 2), dtype=np.float32)


def build_depth_point_grid(n_per_side: int, depth_map: torch.tensor) -> np.ndarray:
    """Generates point grid based on depth maps."""
    sample_points = []
    sample_boxs = []
    h, w = depth_map.shape
    crop_x0 = np.linspace(0, w - 1, n_per_side + 1)[:-1].astype(np.int32)
    crop_w = int(w / len(crop_x0))
    crop_y0 = np.linspace(0, h - 1, n_per_side + 1)[:-1].astype(np.int32)
    crop_h = int(h / len(crop_y0))
    # print(crop_x0,crop_y0,crop_w,crop_h)
    # print(depth_map.shape)
    for x0, y0 in product(crop_x0, crop_y0):
        mean_depth = torch.mean(depth_map[y0:min(y0 + crop_h, h), x0:min(x0 + crop_w, w)])
        sample_num = int(mean_depth)
        if sample_num > 20:
            sample_num = 20
        elif sample_num < 1:
            sample_num = 1
        # print('mean_depth(',x0,y0,')(',x0+crop_w,y0+crop_h,")=",mean_depth)
        offset_x = crop_w / (2 * sample_num)
        offset_y = crop_h / (2 * sample_num)
        points_axis_x = np.linspace(x0 + offset_x, x0 + crop_w - offset_x, sample_num)
        points_axis_y = np.linspace(y0 + offset_y, y0 + crop_h - offset_y, sample_num)
        points_x = np.tile(points_axis_x[None, :], (sample_num, 1))
        points_y = np.tile(points_axis_y[:, None], (1, sample_num))
        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)  # n*n,2
        sample_points.append(points)
        sample_boxs.append(np.array([x0 / w, y0 / h, (x0 + crop_w) / w, (y0 + crop_h) / h]))

    sample_points_concat = np.concatenate(sample_points, axis=0)  # N_points,2
    sample_boxs_concat = np.stack(sample_boxs, axis=0)  # N_boxs,4
    points_scale = np.array(depth_map.shape)[None, ::-1]  # 1,2
    # print('points_scale=',points_scale)
    sample_points_concat = sample_points_concat / points_scale
    # print(sample_points_concat.shape)
    return sample_points_concat, sample_boxs_concat

def build_all_layer_depth_point_grids( # Generate normalized grid points for each level
    n_per_side: int, n_layers: int, scale_per_layer: int, depth_map: torch.tensor
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    boxs_by_layer=[]
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points,box=build_depth_point_grid(n_points, depth_map)
        points_by_layer.append(points)
        boxs_by_layer.append(box)
    return points_by_layer,boxs_by_layer # list(array(n_points,2)) list(array(n_boxs,4))