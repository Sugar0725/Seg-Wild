import torch
from cv2.detail import CameraParams

from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state,PILtoTorch
from argparse import ArgumentParser,Namespace
from arguments import ModelParams, PipelineParams, get_combined_args,args_init
from gaussian_renderer import GaussianModel
import copy,pickle,time
from utils.general_utils import *
import imageio
import torch.nn as nn
import sklearn
import sklearn.decomposition
from models.networks import CNN_decoder, MLP_encoder
import torch.nn.functional as F
from utils.graphics_utils import getWorld2View2
import cv2
from matplotlib import pyplot as plt

def generate_mask(image_path):
    # 读取图像（BGR 格式）
    img = cv2.imread(image_path)

    # 转换为 RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 设定白色阈值（假设 RGB 值接近 255 的像素为白色）
    lower_white = np.array([200, 200, 200], dtype=np.uint8)  # 下界（较宽松）
    upper_white = np.array([255, 255, 255], dtype=np.uint8)  # 上界（纯白）

    # 生成 mask（白色区域为 0，非白色区域为 1）
    mask = cv2.inRange(img_rgb, lower_white, upper_white)
    mask = 1 - (mask // 255)  # 反转，使白色为 0，其他颜色为 1

    return mask

def render_single_view(args, dataset: ModelParams, iteration: int, pipeline: PipelineParams, camera_id):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, args)
        # iteration=1
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  #

        dataset.white_background = True
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.set_eval(True)
        train_camera = scene.getTrainCameras()[camera_id]

        render_pkg = render(train_camera, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        # 可视化渲染结果
        plt.imshow(rendering.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        save_rendering_path = os.path.join(args.model_path, "rendering.png")
        os.makedirs(os.path.dirname(save_rendering_path), exist_ok=True)  # 确保目录存在
        rendering = rendering.clamp(0, 1)  # 限制数据范围
        plt.imsave(save_rendering_path, rendering.permute(1, 2, 0).detach().cpu().numpy())
        print(f"渲染结果已保存至: {save_rendering_path}")

        save_mask_path = os.path.join(args.model_path, "mask.png")
        os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)  # 确保目录存在
        mask = generate_mask(save_rendering_path) * 255
        cv2.imwrite(save_mask_path, mask)
        print(f"mask结果已保存至: {save_mask_path}")




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--render_interpolate", action="store_true", default=False)

    parser.add_argument("--render_multiview_vedio", action="store_true", default=False)

    parser.add_argument("--video", action="store_true")  ###
    parser.add_argument("--novel_video", action="store_true")  ###
    parser.add_argument("--novel_view", action="store_true")  ###
    parser.add_argument("--multi_interpolate", action="store_true")  ###
    parser.add_argument("--num_views", default=100, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    camera_id = 22
    safe_state(args.quiet)
    render_single_view(args, model.extract(args), args.iteration, pipeline.extract(args), camera_id)