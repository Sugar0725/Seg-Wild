import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from utils.scale_utils import (calculate_scale, load_model, gen_sky_mask, init_scene, gen_depth_map, auto_divide_layers,
                               compute_cell_depth_values, allocate_sampling_points, distribute_samples,
                               generate_point_grids, build_all_layer_depth_point_grids)
from arguments import ModelParams, PipelineParams, get_combined_args,args_init
from utils.general_utils import *
from gaussian_renderer import GaussianModel
from scene import Scene
import matplotlib.pyplot as plt


def visualize_masks(img, masks):
    """
    可视化 masks 并覆盖在原图 img 上
    """
    img_h, img_w, _ = img.shape  # 获取原图尺寸
    mask_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)  # 创建一个全黑的彩色 mask 层

    # 确保 masks 是 NumPy 格式
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()  # 转换为 NumPy 数组

    resized_masks = np.zeros((masks.shape[0], img_h, img_w), dtype=np.uint8)
    for i in range(masks.shape[0]):
        resized_masks[i] = cv2.resize(masks[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # 遍历所有 mask 并绘制不同颜色
    for i in range(resized_masks.shape[0]):  # 遍历每个 mask
        mask = resized_masks[i]  # 获取 mask 的二值数组
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # 生成随机颜色

        # 只在 mask 为 1 的地方填充颜色
        mask_overlay[mask > 0] = color

        # 叠加到原图
    alpha = 0.5  # 透明度
    overlay = cv2.addWeighted(img, 1 - alpha, mask_overlay, alpha, 0)

    # 显示结果
    cv2.imshow("Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--image_root", default='./outputs/360_v2/garden/', type=str)
    parser.add_argument("--data_root", default='./data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default="4", type=str)
    parser.add_argument("--use_grid_sam", action='store_true', default=False)

    args = get_combined_args(parser)
    camera_id = 0
    safe_state(args.quiet)

    IMAGE_DIR = os.path.join(args.image_root, fr'full/train/ours_{args.iterations}/renders_intrinsic')
    DATA_DIR = args.data_root

    # 确定初始尺度，之后根据密度分配采样点
    if not args.use_grid_sam:
        scale = calculate_scale(DATA_DIR, target_min=6, target_max=28)
    else:
        scale = calculate_scale(DATA_DIR, target_min=4, target_max=8)
        # scale = calculate_scale(DATA_DIR, target_min=10, target_max=16)

    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(DATA_DIR, 'sam_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    source_namelist = os.listdir(IMAGE_DIR)
    list_count = 0

    print("Extracting SAM segment everything masks...")
    # 预加载模型
    MODEL = load_model()
    print("Succeed to load the DeepLab_v3_plus model")
    # 加载高斯场景
    scene, gaussians = init_scene(args, model.extract(args), args.iteration)
    for idx, path in enumerate(tqdm(os.listdir(IMAGE_DIR)), start=0):
        # if idx < 10:  # 跳过前10张，从第11张开始
        #     idx += 1
        #     continue
        # 如果使用nerf_llff_data数据集生成的seg mask的pt文件名字可能会和下采样之后的images_4文件夹名字一致，
        # 在之后的train_contrastive_features.py中会出现错误，所以要把pt文件名名字改成和images文件夹名字一样
        name = path.split('.')[0]
        pt_path = os.path.join(OUTPUT_DIR, name + '.pt')
        # 查询是否已存在对应的.pt文件
        if os.path.exists(pt_path):
            print(f"Skipping {name}, already processed.")
            list_count += 1
            continue

        img = cv2.imread(os.path.join(IMAGE_DIR, path))

        if args.use_grid_sam:
            # 1.生成天空遮罩
            sky_mask = gen_sky_mask(os.path.join(IMAGE_DIR, path), MODEL, confidence_threshold=0.6)
            # 2.计算深度图
            ori_map, norm_depth = gen_depth_map(scene, gaussians, idx)
            depth_map = norm_depth * (ori_map.max() * 0.3)
            # 3.利用sky_mask过滤深度图
            depth_map[sky_mask == 0] = 0
            # 1. 计算最小值和最大值
            min_val = depth_map.min().item()
            max_val = depth_map.max().item()
            print(f"Depth Map Min: {min_val}, Max: {max_val}")
            # # 4.把图像分割成小块，根据深度和密度分配采样点
            # # 计算分层规则，得到每一层每个小块的像素宽高
            # first_layer, second_layer, third_layer, depth_map_padded = auto_divide_layers(depth_map)
            # # 根据每层分层规则得到每个小块区域，在区域中根据像素深度分配相应的提示点
            # # 计算第一层单元格的深度值
            # first_layer_depths = compute_cell_depth_values(first_layer)
            # # 分配采样点（根据初始尺度scale）
            # first_layer_samples = allocate_sampling_points(first_layer_depths, int(scale[idx] * scale[idx] / 12))
            # # 第二层
            # second_layer_depths = compute_cell_depth_values(second_layer)
            # second_layer_samples = distribute_samples(first_layer_samples, second_layer_depths)
            # print("Total second layer samples:", second_layer_samples.sum().item())
            # # 第三层
            # third_layer_depths = compute_cell_depth_values(third_layer)
            # third_layer_samples = distribute_samples(second_layer_samples, third_layer_depths)
            # print("Total third layer samples:", third_layer_samples.sum().item())
            # # 生成point_grids
            # depth_sample_points = generate_point_grids(third_layer_samples, image_w=img.shape[1], image_h=img.shape[0])
            pps_depth = scale[idx]
            nlayers_depth = 0       # 生成的深度层数，即划分的层级数, 网格的密度会随 scale_per_layer 变化
            scale_per_layer = 1     # 每层缩放因子。每往更深一层，网格点数减少的比例为 1/scale_per_layer
            depth_sample_points, _ = build_all_layer_depth_point_grids(pps_depth, nlayers_depth,
                                                                        scale_per_layer, depth_map)
            #=======================================可视化提示点分布============================================
            # # # 复制原图，避免修改原始数据
            # print(depth_sample_points[0].shape)
            # image_with_points = img.copy()
            # # 绘制点
            # for x, y in depth_sample_points[0]:
            #     cv2.circle(image_with_points, (int(x * img.shape[1]), int(y * img.shape[0])), radius=2, color=(0, 0, 255), thickness=-1)  # 红色小圆点
            #
            # # 显示图像
            # cv2.imshow("Depth Sample Points", image_with_points)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #===============================================================================================
            # 5.最终生成用于SAM分割的prompt_grid
            # generic params
            scale_per_layer = 1
            overlap_ratio = 512 / 1500
            nlayers_depth = 0
            # mask_generator = SamAutomaticMaskGenerator(
            #     model=sam,
            #     points_per_side=None,  ### default
            #     point_grids=depth_sample_points,  ### depth based
            #     pred_iou_thresh=0.7,
            #     box_nms_thresh=0.7,
            #     stability_score_thresh=0.85,
            #     crop_n_layers=nlayers_depth,
            #     crop_n_points_downscale_factor=scale_per_layer,
            #     min_mask_region_area=100,
            #     crop_overlap_ratio=overlap_ratio
            # )
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=0,
                points_per_side=None,
                point_grids=depth_sample_points
            )
            masks = mask_generator.generate(img)
        else:
            # Default settings of SAM
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=0,
                points_per_side=scale[idx]
            )
            masks = mask_generator.generate(img)
        ###
        feature_dim = 64
        img_h, img_w, _ = img.shape
        if img_h > img_w:
            h = feature_dim
            w = int(feature_dim * img_w / img_h + 0.5)
        else:
            w = feature_dim
            h = int(feature_dim * img_h / img_w + 0.5)

        mask_list = []
        if len(masks) == 0:
            masks = torch.ones(1, img_h, img_w).to('cuda')  # 全1填充
        else:
            for m in masks:
                m_score = torch.from_numpy(m['segmentation']).float()[None, None, :, :].to('cuda')
                m_score = torch.nn.functional.interpolate(
                    m_score, size=(img_h, img_w), mode='bilinear', align_corners=False
                ).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                mask_list.append(m_score)

            masks = torch.stack(mask_list, dim=0)
        #=======================================可视化masks=================================================
        # 调用函数
        visualize_masks(img, masks)
        #==================================================================================================
        # # 在保存pt文件时使用的name和在image文件中名字一致，而不是和image_4中名字一致
        # name = source_namelist[list_count].split('.')[0]
        # torch.save(masks, os.path.join(OUTPUT_DIR, name + '.pt'))
        # list_count += 1