#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
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


#

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature

class PCAFeatureVisualizer:
    def __init__(self):
        self.pca = sklearn.decomposition.PCA(3, random_state=42)
        self.mean = None
        self.components = None
        self.postprocess_sub = None
        self.postprocess_div = None
        self.fitted = False

    def fit_pca(self, features):
        """
        计算并保存 PCA 的均值、主成分和后处理参数。
        """
        # 确保输入为 4 维张量
        if features.dim() == 3:  # 如果是 [C, H, W]
            features = features.unsqueeze(0)  # 添加批量维度 [1, C, H, W]
        elif features.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, but got shape {features.shape}")

        # 标准化特征并降维
        f_samples = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()
        transformed = self.pca.fit_transform(f_samples)

        # 保存 PCA 模型参数
        self.mean = torch.tensor(f_samples.mean(0)).float().cuda()
        self.components = torch.tensor(self.pca.components_).float().cuda()
        q1, q99 = np.percentile(transformed, [1, 99])
        self.postprocess_sub = q1
        self.postprocess_div = q99 - q1
        self.fitted = True

    def transform_feature(self, feature):
        """
        使用已拟合的 PCA 模型对特征进行变换和可视化。
        """
        # # 确保输入为 4 维张量
        # if feature.dim() == 3:  # 如果是 [C, H, W]
        #     feature = feature.unsqueeze(0)  # 添加批量维度 [1, C, H, W]
        # elif feature.dim() != 4:
        #     raise ValueError(f"Expected 3D or 4D tensor, but got shape {feature.shape}")

        if not self.fitted:
            raise ValueError("PCA model is not fitted. Call `fit_pca` with appropriate features first.")

        # 标准化并应用已保存的 PCA 模型
        fmap = nn.functional.normalize(feature[None, :, :, :], dim=1)
        vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - self.mean[None, :]) @ self.components.T
        vis_feature = (vis_feature - self.postprocess_sub) / self.postprocess_div
        vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
        return vis_feature


def render_interpolate(model_path, name, iteration, views, gaussians, pipeline, background,select_idxs=None):
    if args.scene_name=="brandenburg":
        select_idxs=[0]#
    elif args.scene_name=="sacre":
        select_idxs=[29]
    elif args.scene_name=="trevi":
        select_idxs=[55]
    elif args.scene_name == "taj":
        select_idxs = [80]
        
    render_path = os.path.join(model_path, name,"ours_{}".format(iteration), f"intrinsic_dynamic_interpolate")
    render_path_gt = os.path.join(model_path, name,"ours_{}".format(iteration), f"intrinsic_dynamic_interpolate","refer")
    makedirs(render_path, exist_ok=True)
    makedirs(render_path_gt, exist_ok=True)
    inter_weights=[i*0.1 for i in range(0,21)]
    select_views=[views[i] for i in select_idxs]
    for idx, view in enumerate(tqdm(select_views, desc="Rendering progress")):
        
        torchvision.utils.save_image(view.original_image, os.path.join(render_path_gt, f"{select_idxs[idx]}_{view.colmap_id}" + ".png"))
        sub_s2d_inter_path=os.path.join(render_path,f"{select_idxs[idx]}_{view.colmap_id}")
        makedirs(sub_s2d_inter_path, exist_ok=True)
        for inter_weight in inter_weights:
            gaussians.colornet_inter_weight=inter_weight
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(sub_s2d_inter_path, f"{idx}_{inter_weight:.2f}" + ".png"))
    gaussians.colornet_inter_weight=1.0


def render_multiview_vedio(model_path,name, train_views,test_views, gaussians, pipeline, background,args):
    if args.scene_name=="brandenburg":
        format_idx=11#4
        select_view_id=[12, 59, 305]
        length_view=90*2
        appear_idxs=[313,78]#
        name="train"
        view_appears=[train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[0,1,2,3,4,5,7,8,9]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views=[train_views[i] for i in select_view_id]
    elif args.scene_name=="sacre":
        format_idx=38 #
        select_view_id=[753,657,595,181,699,]#700
        length_view=45*2
        
        appear_idxs=[350,76]
        name="train"
        view_appears=[train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[6,12,15,17]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views=[train_views[i] for i in select_view_id]
    elif args.scene_name=="trevi":
        format_idx=17 
        select_view_id=[408,303,79,893,395,281]#700
        length_view=45*2
        
        appear_idxs=[317,495]
        
        name="train"
        view_appears=[train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[0,2,3,8,9,11]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views=[train_views[i] for i in select_view_id]
        
    for vid,view_appear in enumerate(tqdm(view_appears, desc="Rendering progress")):
        view_appear.image_height,view_appear.image_width=train_views[format_idx].image_height,train_views[format_idx].image_width
        view_appear.FoVx,view_appear.FoVy=train_views[format_idx].FoVx,train_views[format_idx].FoVy
        appear_idx=appear_idxs[vid]
        generated_views=generate_multi_views(views,view_appear,length=length_view)
        render_path = os.path.join(model_path,"demos" ,f"multiview_vedio",f"{name}_{appear_idx}_{view_appear.colmap_id}") #
        makedirs(render_path, exist_ok=True)
        
        render_video_out = imageio.get_writer(f'{render_path}/000_mv_{name}_{appear_idx}_{view_appear.colmap_id}' + '.mp4', mode='I', fps=30,codec='libx264',quality=10.0)#
        rendering = render(view_appear, gaussians, pipe=pipeline, bg_color=background,store_cache=True)["render"]
        for idx, view in enumerate(tqdm(generated_views, desc="Rendering progress")):
            view.camera_center=view_appear.camera_center
            rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
            render_video_out.append_data(rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
            img_np=save_image(rendering, os.path.join(render_path, f"{name}_{appear_idx}_"+'{0:05d}'.format(idx) + ".png"))
            
            render_video_out.append_data(img_np)

        render_video_out.close()
       
 

def render_lego(model_path, name, iteration, views,view0, gaussians, pipeline, background):
    
    render_path = os.path.join(model_path, name,"ours_{}".format(iteration), f"renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    rendering = render(view0, gaussians, pipe=pipeline, bg_color=background,store_cache=True)["render"]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def test_rendering_speed( views, gaussians, pipeline,background,use_cache=False):
    views=copy.deepcopy(views)
    length=100
    # view=views[0]
    for idx in range(length):
        view=views[idx] 
        view.original_image=torch.nn.functional.interpolate(view.original_image.unsqueeze(0),size=(800,800)).squeeze()
        view.image_height,view.image_width=800,800
    if not use_cache:
        rendering = render(views[0], gaussians, pipeline, background)["render"]
        start_time=time.time()
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx]
            rendering = render(view, gaussians, pipeline, background)["render"]
        end_time=time.time()
        
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    else:
        for i in range(100):
            views[i+1].image_height,views[i+1].image_width=view.image_height,view.image_width
        rendering = render(views[0], gaussians, pipeline, background,store_cache=True)["render"]
        start_time=time.time()
        rendering = render(view, gaussians, pipeline, background,store_cache=True)["render"]
        #for idx, view in enumerate(tqdm(views[1:], desc="Rendering progress")):
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx+1]
            rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
        end_time=time.time()
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed using cache:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    
def render_intrinsic(model_path, name, iteration, views, gaussians, pipeline, background,):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_intrinsic")
    makedirs(render_path, exist_ok=True)
    gaussians.colornet_inter_weight=0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]       
    
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    gaussians.colornet_inter_weight=1.0
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,\
    render_multi_view=False, render_s2d_inter=False, render_semantic=False, speedup=False):
    '''
    '''
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    #-------------------------------------------------semantic feature path-------------------------------------------------
    if render_semantic:
        feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
        gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
        saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
        decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
        if speedup:
            gt_feature_map = views[0].semantic_feature.cuda()
            feature_out_dim = gt_feature_map.shape[0]
            feature_in_dim = int(feature_out_dim / 4)
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        makedirs(feature_map_path, exist_ok=True)
        makedirs(gt_feature_map_path, exist_ok=True)
        makedirs(saved_feature_path, exist_ok=True)
    #-------------------------------------------------semantic feature path--------------------------------------------------

    if gaussians.use_features_mask:
        mask_path=os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
        makedirs(mask_path, exist_ok=True)

    if render_multi_view:
        multi_view_path=os.path.join(model_path, name, "ours_{}".format(iteration), "multi_view")
    if render_s2d_inter:
        s2d_inter_path=os.path.join(model_path, name, "ours_{}".format(iteration), "intrinsic_dynamic_interpolate")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    origin_views=copy.deepcopy(views)

    # 初始化PCA可视化模型
    visualizer = PCAFeatureVisualizer()
    visualizer.fit_pca(views[1].semantic_feature.cuda())  # 拟合PCA参数

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        gt_feature_map = view.semantic_feature.cuda()

        if gaussians.use_features_mask:
            tmask=gaussians.features_mask.repeat(1,3,1,1)
            torchvision.utils.save_image(tmask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # ------------------------------------------render semantic feature--------------------------------------------------------
        if render_semantic:
            feature_map = render_pkg["feature_map"]
            feature_map = F.interpolate(feature_map.unsqueeze(0),
                                        size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear',
                                        align_corners=True).squeeze(0)  ###
            if speedup:
                feature_map = cnn_decoder(feature_map)
            feature_map_vis = feature_visualize_saving(feature_map)
            feature_map_vis = visualizer.transform_feature(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            gt_feature_map_vis = visualizer.transform_feature(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            # save feature map
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            torch.save(torch.tensor(feature_map).half(),os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
        # ------------------------------------------render semantic feature--------------------------------------------------------
    
    if render_multi_view:
        #origin_views=copy.deepcopy(views)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            sub_multi_view_path=os.path.join(multi_view_path,f"{idx}")
            makedirs(sub_multi_view_path, exist_ok=True)
            for o_idx,o_view in enumerate(tqdm(origin_views, desc="Rendering progress")):
                rendering = render(view, gaussians, pipeline, background,\
                         other_viewpoint_camera=o_view)["render"]
                torchvision.utils.save_image(rendering, os.path.join(sub_multi_view_path, f"{idx}_{o_idx}" + ".png"))
    if render_s2d_inter and gaussians.color_net_type in ["naive"]:
        
        views=origin_views
        inter_weights=[i*0.1 for i in range(0,21)]
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            sub_s2d_inter_path=os.path.join(s2d_inter_path,f"{idx}")
            makedirs(sub_s2d_inter_path, exist_ok=True)
            for inter_weight in  inter_weights:
                gaussians.colornet_inter_weight=inter_weight
                rendering = render(view, gaussians, pipeline, background)["render"]
                torchvision.utils.save_image(rendering, os.path.join(sub_s2d_inter_path, f"{idx}_{inter_weight:.2f}" + ".png"))
        gaussians.colornet_inter_weight=1.0
        
    return 0

###
def interpolate_matrices(start_matrix, end_matrix, steps):
    # Generate interpolation factors
    interpolation_factors = np.linspace(0, 1, steps)
    # Interpolate between the matrices
    interpolated_matrices = []
    for factor in interpolation_factors:
        interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
        interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)

###
def multi_interpolate_matrices(matrix, num_interpolations):
    interpolated_matrices = []
    for i in range(matrix.shape[0] - 1):
        start_matrix = matrix[i]
        end_matrix = matrix[i + 1]
        for j in range(num_interpolations):
            t = (j + 1) / (num_interpolations + 1)
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)

###
def render_novel_views(model_path, name, iteration, views, gaussians, pipeline, background,
                       speedup, multi_interpolate, num_views):
    if multi_interpolate:
        name = name + "_multi_interpolate"
    # non-edit
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
    # encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))

    if speedup:
        gt_feature_map = views[0].semantic_feature.cuda()
        feature_out_dim = gt_feature_map.shape[0]
        feature_in_dim = int(feature_out_dim / 4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))

    makedirs(render_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(saved_feature_path, exist_ok=True)

    view = views[0]

    # create novel poses
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)
    if not multi_interpolate:
        poses = interpolate_matrices(render_poses[3], render_poses[170], num_views)
    else:
        poses = multi_interpolate_matrices(np.array(render_poses), 2)

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        # mlp encoder
        render_pkg = render(view, gaussians, pipeline, background)

        gt_feature_map = view.semantic_feature.cuda()
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # visualize feature map
        feature_map = render_pkg["feature_map"]
        feature_map = F.interpolate(feature_map.unsqueeze(0),
                                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear',
                                    align_corners=True).squeeze(0)  ###
        if speedup:
            feature_map = cnn_decoder(feature_map)

        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(),os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))


def render_novel_video(model_path, name, iteration, views, gaussians, pipeline, background, speedup):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
    if speedup:
        gt_feature_map = views[0].semantic_feature.cuda()
        feature_out_dim = gt_feature_map.shape[0]
        feature_in_dim = int(feature_out_dim / 4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    # render_poses = [(cam.R, cam.T) for cam in views]
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)

    # create novel poses
    poses = interpolate_matrices(render_poses[0], render_poses[-1], 200)

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()


def render_sets(args,dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                novel_view : bool, novel_video : bool, multi_interpolate : bool, num_views : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,args)
        #iteration=1
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  #

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.set_eval(True)
        #
        if args.scene_name=="lego":
            render_lego(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),scene.getTrainCameras()[0], gaussians, pipeline, background)
            return 
        if not skip_test:
            # test_rendering_speed(scene.getTrainCameras(), gaussians, pipeline, background)
            # if gaussians.color_net_type in ["naive"]:
            #     test_rendering_speed(scene.getTrainCameras(), gaussians, pipeline, background,use_cache=True)
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, \
                background,render_multi_view=True,render_s2d_inter=True, render_semantic=True, speedup=dataset.speedup)
            
            
        if not skip_train:
            train_cameras=scene.getTrainCameras()
            render_set(dataset.model_path, "train", scene.loaded_iter, train_cameras, gaussians, pipeline, background, render_semantic=True, speedup=dataset.speedup)
            
            if gaussians.color_net_type in ["naive"]:
                render_intrinsic(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if novel_view:
            render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians,
                               pipeline, background,dataset.speedup, multi_interpolate, num_views)

        if novel_video:
            render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(),
                               gaussians, pipeline, background, dataset.speedup)

        if args.render_multiview_vedio:
            render_multiview_vedio(dataset.model_path,"train", scene.getTrainCameras(),scene.getTestCameras(), gaussians, pipeline, background,args)
        if args.render_interpolate:
            #appearance tuning
            render_interpolate(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


        gaussians.set_eval(False)


def render_depth(model_path, name, iteration, views, gaussians, pipeline, background, speedup=False):
    '''
    '''
    render_opacity = True
    if render_opacity:
        opacity_path = os.path.join(model_path, name, "ours_{}".format(iteration), "opacity")
        ori_opacity = gaussians.get_opacity_dealed
        opacity = torch.where(ori_opacity < 0.5, torch.tensor(0.0, device=ori_opacity.device), torch.tensor(1.0, device=ori_opacity.device))
        # 定义 Logit 函数
        def safe_logit(x, epsilon=1e-6):
            x = torch.clamp(x, min=0, max=1 - epsilon)  # 避免 0 和 1 的输入问题
            return torch.log(x / (1 - x))
        new_opacity = safe_logit(opacity)
        gaussians._opacity_dealed = new_opacity
        gaussians._opacity = new_opacity
        opacity_dealed = gaussians.get_opacity_dealed
        makedirs(opacity_path, exist_ok=True)
    else:
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        makedirs(depth_path, exist_ok=True)

    origin_views = copy.deepcopy(views)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if render_opacity:
            render_pkg = render(view, gaussians, pipeline, background)
            opacity_img = render_pkg["render"]
            plt.imshow(opacity_img.permute([1, 2, 0]).detach().cpu())
            plt.show()
            print(opacity_img.shape)

        else:
            render_pkg = render(view, gaussians, pipeline, background)
            depth_map = render_pkg["depth"]
            # 移除批次维度，变为 [H, W]
            depth_map = depth_map.squeeze(0).cpu().numpy()
            # 归一化深度图到 [0, 255] 区间
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255
            depth_map_normalized = depth_map_normalized.astype(np.uint8)
            # 转换为 PIL 图像
            depth_image = Image.fromarray(depth_map_normalized, mode='L')
            # 保存图像
            depth_image.save(os.path.join(depth_path, '{0:05d}'.format(idx) + "_depth.png"))
        # ------------------------------------------render semantic feature--------------------------------------------------------
    return 0

def render_depth_sets(args,dataset : ModelParams, iteration : int, pipeline : PipelineParams,
                      skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,args)
        #iteration=1
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  #

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.set_eval(True)
        if not skip_test:
            test_rendering_speed(scene.getTrainCameras(), gaussians, pipeline, background)
            if gaussians.color_net_type in ["naive"]:
                test_rendering_speed(scene.getTrainCameras(), gaussians, pipeline, background, use_cache=True)

            render_depth(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                         background, speedup=dataset.speedup)

        if not skip_train:
            train_cameras = scene.getTrainCameras()
            render_depth(dataset.model_path, "train", scene.loaded_iter, train_cameras, gaussians, pipeline, background,
                       speedup=dataset.speedup)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--render_interpolate", action="store_true",default=False)

    parser.add_argument("--render_multiview_vedio", action="store_true",default=False)

    parser.add_argument("--video", action="store_true")  ###
    parser.add_argument("--novel_video", action="store_true")  ###
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--multi_interpolate", action="store_true") ###
    parser.add_argument("--num_views", default=7, type=int)

    parser.add_argument("--render_depth", action="store_true", default=False)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    safe_state(args.quiet)
    if not args.render_depth:
        render_sets(args,model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                    args.novel_view, args.novel_video, args.multi_interpolate, args.num_views)
    elif args.render_depth:
        render_depth_sets(args,model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
