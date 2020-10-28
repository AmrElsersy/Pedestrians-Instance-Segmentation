import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class PennnFudanDataset(Dataset):
    def __init__(self,root, transform=None):

        self.transform = transform
        self.root = root

        # root paths
        self.rootImages = os.path.join(self.root, "PNGImages")
        self.rootMasks = os.path.join(self.root, "PedMasks")
        # self.rootAnnotation = os.path.join(self.root, "Annotation")

        # list of data paths
        self.imagesPaths = sorted( os.listdir(self.rootImages) )
        self.masksPaths  = sorted( os.listdir(self.rootMasks) )
        # self.annotationPaths  = sorted( os.listdir(self.rootAnnotation))

        self.imagesPaths = [ os.path.join(self.rootImages, image) for image in self.imagesPaths ]
        self.masksPaths = [ os.path.join(self.rootMasks, mask) for mask in self.masksPaths ]


    def __getitem__(self, index):

        # load image & mask
        image = Image.open(self.imagesPaths[index])
        mask  = Image.open(self.masksPaths[index])

        image = image.convert("RGB")

        # We get the boxes from the masks instead of reading it from a CSV file

        # get list of object IDs (Pedestrians in the mask)
        # ex: if mask has 3 people in it, IDs = [0, 1, 2, 3] ... 0 for background and 1,2,3 for each pedestrian
        IDs = np.unique(np.array(mask))
        # remove the background ID
        IDs = IDs[1:]

        # transpose it to (N,1,1) to be similar to a column vector
        IDs = IDs.reshape(-1,1,1)

        # extract each mask from the IDs 
        masks = np.array(mask) == IDs

        # N Boxes
        N = len(IDs)

        boxes = []
        # area for each box
        area = []

        for i in range(N):
            # where gets the pixels where the mask = True (mask is a 2D Array of true and false , 
            # true at the pixels that is 1 as an indication of the mask & 0 for background)
            mask_pixels = np.where(masks[i])

            # extract the box from the min & max of these points
            # first dim is y , second dim is x
            xmin = np.min(mask_pixels[1])
            xmax = np.max(mask_pixels[1])
            ymin = np.min(mask_pixels[0])
            ymax = np.max(mask_pixels[0])

            boxes.append([xmin, ymin, xmax, ymax])
            area.append((ymax-ymin) * (xmax-xmin))

        # convert 2D List to 2D Tensor (this is not numpy array)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)

        # labels for each box
        # there is only 1 class (pedestrian) so it will always be 1 (if multiple classes, so we will assign 1,2,3 ... etc to each one)
        labels = torch.ones((N,), dtype=torch.int64)

        # image_id requirement for model, index is unique for every image
        image_id = torch.tensor([index], dtype=torch.int64)

        # instances with iscrowd=True will be ignored during evaluation.
        # set all = False (zeros)
        iscrowd = torch.zeros((N,), dtype=torch.uint8)

        # convert masks to tensor (model requirement)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # print("image size=", image.size)
        # print("mask size=", mask.size)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = area
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.imagesPaths)


