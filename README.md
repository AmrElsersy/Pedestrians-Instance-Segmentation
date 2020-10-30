# Pedestrians Instance Segmentation with Mask R-CNN using Pytorch

Pedestrians detection and segmentation with Mask R-CNN using transfer learning on PennFudanPed Dataset
Implemented in **Pytorch**

![Screenshot from 2020-10-28 14-59-52](https://user-images.githubusercontent.com/35613645/97443320-8fb59280-1933-11eb-8fb2-83a436e7b0ab.png)


## Dataset

PennFundanPed Dataset consist of **170 images** of pedestrians and segmentation **mask** for each image

Dataset Link: (https://www.cis.upenn.edu/~jshi/ped_html/)


## Neural Network 

* Used a pretrained Mask R-CNN

* Transfer learning on Faster R-CNN used by Mask R-CNN to modify the fully connected layers responsiple for classfication and bounding boxes regression

* Transfer learning on last layer of the mask detection of Mask R-CNN

* Fixed feature extractor 


## Examples

![Screenshot from 2020-10-28 15-17-50](https://user-images.githubusercontent.com/35613645/97443331-9217ec80-1933-11eb-9a5f-02ceeb9cfabe.png)


### Semantic Segmentation (using same color for each class)

![z3](https://user-images.githubusercontent.com/35613645/97431419-eff00880-1922-11eb-8c39-43c3d1f31347.png)

![z2](https://user-images.githubusercontent.com/35613645/97433487-4579e480-1926-11eb-8582-7e54fc7b99ac.png)

![z1](https://user-images.githubusercontent.com/35613645/97433494-4874d500-1926-11eb-8df6-f37c965be25f.png)

