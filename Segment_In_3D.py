from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,args_init
import sys
from scene import Scene, GaussianModel
import os
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.decomposition
import torch
from gaussian_renderer import render
import torch.nn as nn
from models.networks import CNN_decoder, MLP_encoder
from render import feature_visualize_saving
import cv2
import pytorch3d.ops as ops
import hdbscan
import random
from utils.point_utils import mask_pcd_2d, project_gs_idx, project_gs_name, get_global_idx, get_filename_by_id, Spiky_3DGaussians_Cutter
import glob

# # 设置新的工作路径
# new_path = 'H:\PycharmProject\Gaussian-Wild-main-feature'
# os.chdir(new_path)
# print("更改后的工作路径:", os.getcwd())

parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*5000 for i in range(1,20)])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
parser.add_argument("--quiet", action="store_true")

parser.add_argument("--render_after_train",  action='store_true', default=True)
parser.add_argument("--metrics_after_train",  action='store_true', default=True)
parser.add_argument("--data_perturb", nargs="+", type=str, default=[])
parser.add_argument("--segment_name", type=str, default=None)
parser.add_argument("--sam_checkpoint_path", default="./dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
parser.add_argument('--ref_img', type=int, default=0)
parser.add_argument('--rend_img', type=int, default=0)

args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)
args = args_init.argument_init(args)
print("Optimizing " + args.model_path)
args.speedup = True

# Set SAM Predictor
import sys
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = args.sam_checkpoint_path
sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
sam.to("cuda")

predictor = SamPredictor(sam)
print("succeed to load sam_model")

# 读取图像文件
image_id = args.ref_img
tsv_path = glob.glob(os.path.join(os.path.dirname(args.source_path.rstrip("/")), "*.tsv"))[0]
global_idx = get_global_idx(tsv_path, image_id)
image_name = get_filename_by_id(tsv_path, global_idx)
idx = int(float(global_idx))
image_path = os.path.join(args.source_path, "images", image_name)
gaussians = GaussianModel(3,args)

scene = Scene(lp.extract(args), gaussians, shuffle=False)
scale_image_path = os.path.join(args.model_path, fr"train/ours_{args.iterations}/renders_intrinsic/{str(image_id).zfill(5)}.png")
feature_map = torch.load(os.path.join(args.model_path, fr"train/ours_{args.iterations}/saved_feature/{str(image_id).zfill(5)}_fmap_CxHxW.pt"))
gaussians.load_ckpt_ply(os.path.join(args.model_path, fr"ckpts_point_cloud/iteration_{args.iterations}/point_cloud.ply"))
sem_features = gaussians.get_semantic_feature

original_image = cv2.imread(image_path)

image_scale = True
if image_scale:
    # 获取原始图像的尺寸
    original_h, original_w = original_image.shape[:2]
    # 使用 OpenCV 进行插值，缩放 scale_image 到与 original_image 相同的分辨率
    scale_image = cv2.imread(scale_image_path)
    original_image = cv2.resize(scale_image, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

if hasattr(feature_map, 'is_cuda'):
    feature_map = feature_map
else:
    feature_map = torch.from_numpy(feature_map)
feature_map = feature_map.permute([1, 2, 0])
plt.imshow(original_image)

feature_height = feature_map.shape[0]
feature_width = feature_map.shape[1]
image_height = original_image.shape[0]
image_width = original_image.shape[1]

#构造CNN解码器并加载已经训练的参数
if args.speedup:
    views = scene.getTrainCameras()
    gt_feature_map = views[0].semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]
    feature_in_dim = int(feature_out_dim / 4)
    cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
    decoder_ckpt_path = os.path.join(args.model_path, fr"decoder_chkpnt{args.iterations}.pth")
    cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))

if args.speedup:
    scale_conditioned_point_features = cnn_decoder(sem_features.permute([2,0,1]))
else:
    scale_conditioned_point_features = sem_features
scale_conditioned_point_features = scale_conditioned_point_features.permute([1,2,0]) #256维

normed_scale_conditioned_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)
normed_features = torch.nn.functional.normalize(feature_map.to(torch.float32), dim=-1, p=2)

bg_color = [1,1,1]
# 使用可视化窗口记录提示点位置--------------------------------------------------------------------
# 初始化全局变量
query_indices = []  # 用于存储点击的坐标
image = None  # 图像数据
logits = None
# Initialize the list of keypoints
keypoints = []
mode = 1
# 复制图像以便绘制标记
image = original_image.copy()
if predictor is not None:
    predictor.set_image(image)
# 设置鼠标回调函数，用于捕捉点击坐标
sam_mask = None
def annotate_keypoints(event, x, y, flags, param):
    global query_indices, mode, image, logits, keypoints, sam_mask

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击事件
        # 添加点击坐标和模式
        keypoints.append((x, y, mode))  # OpenCV 中是 (y, x) 而非 (x, y)

        if predictor is not None:
            # 运行 SAM 模型
            input_point = np.array([pts[:2] for pts in keypoints])  # 提取坐标
            input_label = np.array([pts[2] for pts in keypoints])  # 提取模式标签
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=logits,
                multimask_output=False,
            )
            sam_mask = masks[0]

            # # 从 mask 中随机选择点
            # mask_coords = np.argwhere(sam_mask)  # 获取 mask 中所有 True 的坐标 (y, x)
            # num_points = max(1, int(len(mask_coords) * 0.001))  # 按比例选取，至少一个点
            # selected_coords = random.sample(list(mask_coords), num_points)  # 随机选择点
            # query_indices = []
            # for coord in selected_coords:
            #     query_indices.append((coord[0], coord[1]))  # 以 (y, x) 添加点
            query_indices.append((y, x))


            # 生成带颜色的遮罩
            color_mask = (np.random.random(3) * 255).astype(np.uint8)
            colored_mask = np.repeat(sam_mask[:, :, np.newaxis], 3, axis=2) * color_mask
            image = cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)
        else:
            image = original_image.copy()

        colors = [(0, 0, 255), (0, 255, 0)]
        # 在图像上绘制点击点
        for y, x, m in keypoints:
            cv2.circle(image, (y, x), 3, colors[m], -1)
        cv2.imshow("2D Annotator", image)

    elif event == cv2.EVENT_RBUTTONDOWN:  # 鼠标右键点击事件
        if mode == 1:
            mode = 0
        elif mode == 0:
            mode = 1

# 创建窗口并设置鼠标回调函数
cv2.imshow("2D Annotator", image)
cv2.setMouseCallback("2D Annotator", annotate_keypoints)
# 等待用户关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
# 输出记录的所有点击点坐标
print("点击的坐标记录为:", query_indices)
for query_index in query_indices:
    plt.scatter(query_index[1], query_index[0], color='red', marker='x')  # 在图像上标记坐标，注意 x 和 y 的顺序
# 显示图像
plt.show()

query_features = []
query_similarities = []  # 存储提示点的相似度
for query_index in query_indices:
    query_index_scaled = (
        int(query_index[0] / image_height * feature_height),
        int(query_index[1] / image_width * feature_width)
    )
    query_feature = normed_features[query_index_scaled[0], query_index_scaled[1]]
    query_features.append(query_feature)

similarities = torch.einsum('C,NC->N', query_features[0].cuda(), normed_scale_conditioned_point_features.squeeze(1))
# 平均相似度---------------------------------------------------------------------------------------------------------------
# 计算每个提示点与所有点的
# ---------------------------------------------相似度
# similarities = torch.zeros(normed_scale_conditioned_point_features.shape[0], device="cuda")
# for query_feature in query_features:
#     similarities += torch.einsum('C,NC->N', query_feature.cuda(), normed_scale_conditioned_point_features.squeeze(1))
#
# # 对所有提示点的相似度进行平均（或者其他方式处理）
# similarities /= len(query_features)  # 平均相似度--------------------------------------------------------------------------
print(similarities.min(), similarities.max())
similarities[similarities < 0.2] = 0

bg_color = [1,1,1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
cameras = scene.getTrainCameras()
render_idx = args.rend_img
rendered_similarities = render(cameras[image_id], gaussians, pp.extract(args), background, override_color=similarities.unsqueeze(-1).repeat([1,3]))['render']

# plt.imshow(rendered_similarities.permute([1,2,0])[:,:,0].detach().cpu() > 0.6)
plt.imshow(rendered_similarities.permute([1,2,0])[:,:,0].detach().cpu() > 0.2)
plt.show()
#sam的2维mask投影分割
# uv_cam, _, _ = project_gs_idx(gaussians.get_xyz, idx, args.source_path)
uv_cam, _, _ = project_gs_name(gaussians.get_xyz, image_name, args.source_path)
project_mask = mask_pcd_2d(uv_cam, sam_mask)
count_mask = np.count_nonzero(project_mask)
project_mask = torch.tensor(project_mask.astype(np.int32), dtype=torch.int32, device="cuda")

# gaussians = gaussians.segment(project_mask)
# print(gaussians.get_xyz.shape)
# background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
# cameras = scene.getTrainCameras()
# seg_res = render(cameras[image_id], gaussians, pp.extract(args), background)['render']
# plt.imshow(seg_res.permute([1, 2, 0]).detach().cpu())
# plt.show()

# count_similarities = torch.sum(similarities > 0.4)
segment_gaussians = gaussians.segment(((similarities > 0.3) + project_mask) == 2)
# segment_gaussians = gaussians.segment(project_mask == 1)
# segment_gaussians = gaussians.segment((similarities > 0.4) == 1)
# 进行尖锐高斯切割
segment_gaussians = Spiky_3DGaussians_Cutter(segment_gaussians, scene, sam_mask, image_id)
# =============================================================================================================
if args.segment_name is not None:
    segment_point_cloud_path = os.path.join(os.path.dirname(args.model_path), args.segment_name, f"ckpts_point_cloud/iteration_{args.iterations}")
    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(segment_point_cloud_path):
        os.makedirs(segment_point_cloud_path)
    segment_gaussians.save_ckpt_ply(os.path.join(segment_point_cloud_path, "point_cloud.ply"))
# =============================================================================================================
# segment_gaussians = gaussians.segment(similarities > 0.4)
print(segment_gaussians.get_xyz.shape)
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
cameras = scene.getTrainCameras()
# seg_res = render(cameras[image_id], segment_gaussians, pp.extract(args), background)['render']
seg_res = render(cameras[render_idx], segment_gaussians, pp.extract(args), background)['render']

# 将图像数据的维度进行调整，使其符合plt.imshow()要求的格式（通常是(height, width, channels)）
image_data = seg_res.permute([1, 2, 0]).detach().cpu()

# 第一步：将数值范围从0 - 1调整到0 - 255
image_save = (image_data.numpy() * 255).astype(np.uint8)

plt.imshow(seg_res.permute([1, 2, 0]).detach().cpu())
plt.show()

def cluster_points_and_generate_mask(pcd, min_cluster_size=150, metric='euclidean'):
    """
    使用 HDBSCAN 算法对 3D 点云进行聚类，并生成一个掩码。

    参数:
    - points: np.ndarray, 形状为 (N, 3)，表示 N 个 3D 点的坐标。
    - min_cluster_size: int, 最小聚类大小，默认 5。
    - metric: str, 用于计算距离的度量方式，默认 'euclidean'，可以使用其他距离度量如 'manhattan'。

    返回:
    - mask: np.ndarray, 形状为 (N,) 的布尔数组，True 表示主类点，False 表示其他类点。
    - main_cluster_label: int, 主类（最大的类）的标签。
    """
    # 如果输入是 GPU 上的张量（torch.Tensor），将其转换为 NumPy 数组
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()  # 先 detach，再转为 NumPy 数组
    # 使用 HDBSCAN 进行聚类
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, core_dist_n_jobs=1)
    cluster_labels = clusterer.fit_predict(pcd)

    # 找到主类（最大的类），排除噪声类（标签为 -1）
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    # 排除噪声类 (-1) 后的有效标签和计数
    valid_labels = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]

    # 找到出现最多的类
    max_idx = np.argmax(valid_counts)
    main_cluster_label = valid_labels[max_idx]

    # 生成掩码，主类的点为 True，其他类的点为 False
    mask = cluster_labels == main_cluster_label

    return mask, main_cluster_label

# _, _, mask, _ = postprocess_statistical_filtering(segment_gaussians.get_xyz, precomputed_mask=mask, max_time=5)
mask, _ = cluster_points_and_generate_mask(segment_gaussians.get_xyz)
segment_gaussians = segment_gaussians.segment(torch.from_numpy(mask).cuda())
print(segment_gaussians.get_xyz.shape)

bg_color = [1,1,1]
# bg_color = [0 for i in range(FEATURE_DIM)]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
seg_res = render(cameras[render_idx], segment_gaussians, pp.extract(args), background)['render']

# 将图像数据的维度进行调整，使其符合plt.imshow()要求的格式（通常是(height, width, channels)）
image_data = seg_res.permute([1, 2, 0]).detach().cpu()

plt.imshow(image_data)
plt.show()











