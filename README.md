# Seg-Wild: Interactive Segmentation based on 3D Gaussian Splatting for Unconstrained Image Collections
We propose Seg-Wild, a novel interactive 3D segmentation framework tailored for unconstrained photo collections. Built on 3D Gaussian Splatting, Seg-Wild embeds multi-dimensional features into 3D Gaussians and performs interactive segmentation via feature similarity with user-specified targets. To mitigate noise and artifacts in unconstrained data, we introduce the Spiky 3D Gaussian Cutter (SGC) for geometric refinement. We also develop a benchmark to evaluate segmentation quality in in-the-wild scenarios.

## Framework Overview
<p align="center">
  <img src="https://github.com/sugerkiller/Seg-Wild/blob/main/assets/pipline.png" width="90%">
</p>
An overview of our framework. During the reconstruction of in-the-wild scenes, we embed affinity features into 3D Gaussians to construct a 3D feature field. Optimized by the scale-adaptive segmentation module (SASM), the SAM mask promotes feature compactness for improved segmentation. In the segmentation process, we use the prompt points $\mathscr{pp}$ to find the feature embeddings of the reference image $I_i$ and calculate the similarity with the affinity features $af$ to identify the similar 3D Gaussians. The spiky 3D Gaussians cutter (SGC) refines the segmentation results to obtain the final output segmentation.

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


