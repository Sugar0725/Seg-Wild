# Seg-Wild: Interactive Segmentation based on 3D Gaussian Splatting for Unconstrained Image Collections (ACM MM 2025)
We propose Seg-Wild, a novel interactive 3D segmentation framework tailored for unconstrained photo collections. Built on 3D Gaussian Splatting, Seg-Wild embeds multi-dimensional features into 3D Gaussians and performs interactive segmentation via feature similarity with user-specified targets. To mitigate noise and artifacts in unconstrained data, we introduce the Spiky 3D Gaussian Cutter (SGC) for geometric refinement. We also develop a benchmark to evaluate segmentation quality in in-the-wild scenarios.

## Framework Overview
<p align="center">
  <img src="https://github.com/sugerkiller/Seg-Wild/blob/main/assets/pipline.png" width="90%">
</p>
An overview of our framework. During the reconstruction of in-the-wild scenes, we embed affinity features into 3D Gaussians to construct a 3D feature field. Optimized by the scale-adaptive segmentation module (SASM), the SAM mask promotes feature compactness for improved segmentation. In the segmentation process, we use the prompt points **\mathscr{pp}** to find the feature embeddings of the reference image $I_i$ and calculate the similarity with the affinity features $af$ to identify the similar 3D Gaussians. The spiky 3D Gaussians cutter (SGC) refines the segmentation results to obtain the final output segmentation.

## Environment setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda create --name seg-wild python=3.7
conda activate seg-wild
```
PyTorch (Please check your CUDA version, we used 11.6)
```shell
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu116
```
Other required Python libraries are as follows:
```shell
pytorch3d == 0.7.1
torch_scatter == 2.1.0
hdbscan == 0.8.36
```
Required packages
```shell
pip install -r requirements.txt
```
Submodules
```shell
pip install submodules/diff-gaussian-rasterization-depth # Rasterizer for RGB, n-dim feature, depth
pip install submodules/simple-knn
```

## Checkpoints
### DeepLabV3+ checkpoints
You need to create a folder named ```./assets/checkpoints``` for the DeepLabV3+ checkpoints.
```shell
Seg-Wild
|---assets
    |---checkpoints
        |---best_deeplabv3plus_mobilenet_cityscapes_os16.pth
```
We use ```best_deeplabv3plus_mobilenet_cityscapes_os16.pth``` as the pre-trained model. You can download the pre-trained model from [Google drive](https://drive.google.com/file/d/1LY5U3_3f4cBun0hSgUiRy5H30cIxcBw_/view?usp=drive_link).

### SAM checkpoints
You will also need to create a folder named ```./encoder/sam_encoder/checkpoints``` for the SAM checkpoints.
```shell
Seg-Wild
|---encoders
    |---sam_encoder
        |---checkpoints
            |---sam_vit_h_4b8939.pth
```
You can download the pre-trained model from [Google drive](https://drive.google.com/file/d/1l687LyPCVrhaAlhsEREM5eOYHQw4msy2/view?usp=drive_link).

## Data
### Data Structure
```shell
Seg-Wild
|---data
    |---trevi_fountain
        |---trevi.tsv
        |---dense
            |---images
            |---sparse
            |---stereo
```
The Photo Tourism datasets can be downloaded from [Image Matching: Local Features & Beyond 2020](https://www.cs.ubc.ca/~kmyi/imw2020/data.html).

You can download the ```.tsv``` file from [Google drive](https://drive.google.com/file/d/1k4BoVVhEZKsjBv_YBdLdz_K2hw3vdkZX/view?usp=drive_link) we provide.


## Training, Segmenting and Rendering
As an example, we demonstrate the segmentation of Oceanus in the trevi_fountain scene.
### Data Preprocessing
You need to use the SAM_encoder to extract feature embeddings from the data and save them into the ```./data/DATASET_NAME/dense/sam_embeddings``` folder.

```export_image_embeddings.py``` is located at ```Seg-Wild/encoders/sam_encoder/```
```shell
python export_image_embeddings.py --checkpoint Path_to_SAM_ckpt --model-type vit_h --input data\DATASET_NAME\images --output data\DATASET_NAME\sam_embeddings
```

### Training
```shell
python train.py --source_path data/trevi_fountain/dense/ --scene_name trevi --model_path outputs/trevi/full --eval --resolution 2 --iterations 30000
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

```--source_path``` / -s Path to the source directory containing a PT dataset.

```--scene_name``` Trained model name

```--model_path``` / -m Path where the trained model should be stored (```output/scene_name``` by default).

```--resolution``` / -r Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. If proveided ```0```, use GT feature map's resolution. For all other values, rescales the width to the given number while maintaining image aspect. If proveided ```-2```, use the customized resolution (```utils/camera_utils.py L31```). **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**

</details>

### Extract SAM Mask
We designed a SASM module that can adaptively generate prompt points based on image distances and object distribution information to optimize SAM segmentation. The SAM masks generated by the code ```extract_segment_everything_masks.py``` will be saved in ```./data/DATASET_NAME/dense/sam_masks```.
```shell
python extract_segment_everything_masks.py --image_root ./outputs/trevi --data_root ./data/trevi_fountain/dense --use_grid_sam --model_path outputs/trevi/full --sam_checkpoint_path Path_to_SAM_ckpt --downsample 4 --resolution 2
```

### Final Training
```shell
python train.py --source_path data/trevi_fountain/dense/ --scene_name trevi --model_path outputs/trevi/full --eval --resolution 2 --iterations 30000 --use_sam_masks
```
```--use_sam_masks``` is used to control whether ```sam_masks``` are involved in training.

### Rendering
```shell
python render.py --model_path outputs/trevi/full --resolution 2
```

### Segmenting
```shell
python Segment_In_3D.py --sam_checkpoint_path Path_to_SAM_ckpt --source_path data/trevi_fountain/dense/ --scene_name trevi --model_path outputs/trevi/full --eval --segment_name trevi_test --resolution 2 --iterations 30000 --ref_img 0
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for Segment_In_3D.py</span></summary>

```--segment_name``` specifies the name of the folder where the segmented model will be stored. It is located in the same directory as the ```full``` folder.

```--ref_img``` specifies which image to use as the reference image for segmentation (default is ```0```).

```--rend_img``` represents the ```idx``` of the rendered image after segmentation (default is ```0```).
</details>

## Experiments
### Qualitative Results
<p align="center">
  <img src="https://github.com/sugerkiller/Seg-Wild/blob/main/assets/qualitative.png" width="80%">
</p>

We compare our method with Feature 3DGS, SAGA, GS-W with projection-based segmentation.

### Visualization results of the novel view synthesis
<p align="center">
  <img src="https://github.com/sugerkiller/Seg-Wild/blob/main/assets/novel_view.png" width="70%">
</p>
We perform novel view synthesis on the segmentation and render the results using different camera poses.

### Visualization results of appearance tuning
<p align="center">
  <img src="https://github.com/sugerkiller/Seg-Wild/blob/main/assets/appearance.png" width="70%">
</p>
We perform appearance tuning on the segmented regions by interpolating the appearance weights of two images.


