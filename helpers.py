from PennFudan_Dataset import PennnFudanDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Subset, DataLoader
import os
import numpy as np
import random

# convert the mask into a colored one

def get_coloured_mask(mask):

    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    colors = [[255,0,0]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colors[random.randrange(0,len(colors))]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def intersection_over_union(box1, box2):

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(yi2-yi1, 0) * max(xi2-xi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1]) *  (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) *  (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou

# define our collate to allow data and target with different sizes
# as default collate (which collect the images,targets of the patch) dosn't allow diffirent sizes

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


# get the PennFudanDataset train & test loaders

def get_dataset_loaders(transform, batch_size, test_batch_size, root, split_perecentage):

    # Load Dataset
    dataset = PennnFudanDataset(root, transform)

    # Split dataset into train and test
    n = len(dataset)
    factor_subset = int(split_perecentage * n)

    train_dataset = Subset(dataset, list( range(0, factor_subset) ) )
    test_dataset = Subset(dataset,  list( range(factor_subset, n) ) )

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=my_collate)

    return train_loader, test_loader, dataset

