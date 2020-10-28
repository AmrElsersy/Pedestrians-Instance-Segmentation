from PennFudan_Dataset import PennnFudanDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Subset, DataLoader
from PIL import Image
import os, time, copy, random
import numpy as np
import cv2

# last layer of each architecture for transfer learning
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from helpers import get_dataset_loaders, get_coloured_mask, intersection_over_union

# from Coco_eval import evaluate

def mask_rcnn_transfer_learning(is_finetune : bool):

    mask_RCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # just train the modified layers
    if not is_finetune:
        for param in mask_RCNN.parameters():
            param.requires_grad = False

    # print(mask_RCNN)

    in_features_classes_fc = mask_RCNN.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels

    # same num of conv5_mask output
    hidden_layer = 256

    # num_classes = 0 (background) + 1 (person) === 2
    fastRCNN_TransferLayer = FastRCNNPredictor(in_channels=in_features_classes_fc, num_classes= 2)
    maskRCNN_TransferLayer = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=hidden_layer, num_classes= 2)

    mask_RCNN.roi_heads.box_predictor = fastRCNN_TransferLayer
    mask_RCNN.roi_heads.mask_predictor = maskRCNN_TransferLayer

    # print(mask_RCNN)

    return mask_RCNN


class Pedestrian_Segmentation:
    def __init__(self):

        # Hyperparameters
        self.root = "PennFudanPed"
        self.transform = transforms.Compose([transforms.ToTensor() ])

        self.batch_size = 2
        self.learning_rate = 0.005
        self.epochs = 10

        self.split_dataset_factor = 0.7

        # dataset
        # test_batch_size = 1 for looping over single sample
        self.train_loader, self.test_loader, self.dataset = get_dataset_loaders(self.transform, self.batch_size, 1, self.root, self.split_dataset_factor)

        # model
        self.mask_RCNN = mask_rcnn_transfer_learning(is_finetune=False)

        # parameters of the modified layers via transfer learning
        fast_rcnn_parameters = [ param for param in self.mask_RCNN.roi_heads.box_predictor.parameters()] 
        mask_rcnn_parameters = [ param for param in self.mask_RCNN.roi_heads.mask_predictor.parameters()]
        self.parameters = fast_rcnn_parameters + mask_rcnn_parameters

        # optimizer & lr_scheduler
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=3, gamma=0.1)

        # path to save / load weights
        self.weights_path = "./weights.pth"

        # threshould for bounding box evaluation 
        self.IoU_threshould = 0.5

    def train_one_epoch(self):

        bathes_per_epoch = len(self.dataset) / self.batch_size
        for i, (images, targets) in enumerate(self.train_loader):

            # model in training mode accepts both input and output and return the loss of all types
            loss = self.mask_RCNN(images, targets)

            # Loss = L_cls + L_box + L_mask + L_objectness + L_rpn
            
            L_cls = loss["loss_classifier"]
            L_box = loss["loss_box_reg"]
            L_mask = loss["loss_mask"]
            L_objectness = loss["loss_objectness"]
            L_rpn = loss["loss_rpn_box_reg"]

            L = L_cls + L_box + L_mask + L_objectness + L_rpn

            L.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print("Loss = ", L.item(), " batch = ", i, "/", bathes_per_epoch)


    def train(self):

        # set to training mode
        self.mask_RCNN.train()

        t1 = time.time()

        for epoch in range(self.epochs):
            print (epoch + 1, "/", self.epochs)
            self.train_one_epoch()
            self.lr_scheduler.step()
            

        t2 = time.time() - t1
        print("time = ", t2)

    def eval(self):
        
        # eval mode 
        self.mask_RCNN.eval()

        # # using pycocotools
        # device = torch.device('cpu')
        # evaluate(self.mask_RCNN, self.test_loader, device)
        # return

        mA_Recall = 0
        all_true_boxes = 0

        # Wrong detections (< threshould)
        FalsePositives = []
        # Correct detections 
        TruePositives = []

        with torch.no_grad():
            for i, (image, target) in enumerate(self.test_loader):
                print(f"sample evaluation {i}")                
                # we had 1 sample in the batch (batch size of test loader = 1)
                output = self.mask_RCNN(image)[0]
                target = target[0]
                
                detected_boxes = output["boxes"]
                scores = output["scores"]
                # labels = output["labels"] # not used in case of 1 class

                true_boxes = target["boxes"]
                # flags for already checked true boxes
                checked_true_boxes = [False for i in range(len(true_boxes))]

                for i_box, box in enumerate(detected_boxes):
                    best_IoU = 0
                    best_true_box = -1

                    # get the best IoU with the true boxes
                    for i_true_box, true_box in enumerate(true_boxes):
                        IoU = intersection_over_union(box, true_box)

                        if IoU > best_IoU:
                            best_IoU = IoU
                            best_true_box = i_true_box
                    # ======================================

                    # if the best IoU (best true box fit for that detected box) > threshould
                    # check if the true box is already assigned to another box so it will be wrong (False Positive)
                    # if not assigned -> Correct detection -> True Positive
                    if best_IoU > self.IoU_threshould:
                        if checked_true_boxes[best_true_box]:
                            FalsePositives.append(i_box)
                        else:
                            TruePositives.append(i_box)
                            checked_true_boxes[best_true_box] = True
                    else:
                        FalsePositives.append(i_box)

                
                all_true_boxes += len(true_boxes)

            all_true_positives = len(TruePositives)
            all_false_positives = len(FalsePositives)

            Recall = all_true_positives / all_true_boxes
            Percesion = all_true_positives / (all_false_positives + all_true_positives + 1e-5)

            print(f"Recall = {Recall} & Percesion = {Percesion} ")
            return Recall, Percesion

    def save(self):
        torch.save(self.mask_RCNN.state_dict(), self.weights_path)

    def load(self):
        weights = torch.load(self.weights_path)
        self.mask_RCNN.load_state_dict(weights)

    def detect(self, path):

        transform = torchvision.transforms.ToTensor()

        image = Image.open(path)
        image = transform(image)

        with torch.no_grad():
            self.mask_RCNN.eval()
            output = self.mask_RCNN([image])[0] # [0] because we pass 1 image

            # print(output)

            # convert dark masking into one-hot labeled
            # masks contains 0 and a low gray value, so it will be considered as 0 
            # convert to numpy to deal with it by openCV
            masks = (output["masks"] >= 0.5).squeeze().numpy()    
            boxes = output["boxes"].numpy()

            img = cv2.imread(path)
            original = img

            for i, mask in enumerate( masks ):
                mask = get_coloured_mask(mask)
                mask = mask.reshape(img.shape)

                img = cv2.addWeighted(img, 1, mask, 0.5, 0)

                cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2],boxes[i][3]) , (0,200,0))

            cv2.imshow("original", original)
            cv2.imshow("masked",img)

            cv2.waitKey(0)



model = Pedestrian_Segmentation()
model.load()

# Test
index = 5
root = "PennFudanPed/Test"
paths = sorted(os.listdir("PennFudanPed/Test"))
path = os.path.join(root, paths[index])

model.detect(path)


