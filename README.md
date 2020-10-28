# Pedestrians Segmentation with Mask R-CNN using Pytorch

Pedestrians detection and semantic segmentation with Mask R-CNN using transfer learning on PennFudanPed Dataset
Implemented in **Pytorch**

![z3](https://user-images.githubusercontent.com/35613645/97431419-eff00880-1922-11eb-8c39-43c3d1f31347.png)


## Dataset

PennFundanPed Dataset consist of **170 images** of pedestrians and segmentation **mask** for each image

![Dataset Link](https://www.cis.upenn.edu/~jshi/ped_html/)


## Neural Network 

* Used a pretrained Mask R-CNN

* Transfer learning on Faster R-CNN used by Mask R-CNN to modify the fully connected layers responsiple for classfication and bounding boxes regression

* Transfer learning on last layer of the mask detection of Mask R-CNN

* Fixed feature extractor 


## Examples

![z2](https://user-images.githubusercontent.com/35613645/97433487-4579e480-1926-11eb-8582-7e54fc7b99ac.png)


![z1](https://user-images.githubusercontent.com/35613645/97433494-4874d500-1926-11eb-8df6-f37c965be25f.png)

