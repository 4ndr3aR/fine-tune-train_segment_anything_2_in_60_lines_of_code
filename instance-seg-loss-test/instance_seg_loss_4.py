#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceSegmentationLoss(nn.Module):
    def __init__(self):
        super(InstanceSegmentationLoss, self).__init__()

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss given predicted masks and ground truth masks

        Args:
          pred_mask: Predicted mask as a PyTorch Tensor (B, C, H, W), 
                     where C is the number of classes (including background)
          true_mask: True mask as a PyTorch Tensor (B, H, W), 
                     with integer values representing instance IDs
        Returns:
          loss: The calculated loss as a PyTorch tensor (scalar)
        """

        # One-hot encode the true mask
        num_classes = pred_mask.shape[1]
        print(f'num_classes: {num_classes}')
        true_mask_onehot = F.one_hot(true_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Calculate the IoU for each class
        intersection = (pred_mask * true_mask_onehot).sum(dim=(2, 3))
        union = (pred_mask.sum(dim=(2, 3)) + true_mask_onehot.sum(dim=(2, 3)) - intersection)

        # Avoid division by zero
        iou = intersection / (union + 1e-6)

        # Calculate the loss as the mean of 1 - IoU for all classes (excluding background)
        loss = 1 - iou[:, 1:].mean() 

        return loss

if __name__ == '__main__':
    # Example Usage
    # Define a dummy prediction and ground truth masks
    pred_mask = torch.zeros((128, 128), dtype=torch.int32)
    pred_mask[20:70, 20:70] = 1
    pred_mask[50:90, 60:110] = 2

    true_mask = torch.zeros((128, 128), dtype=torch.int32)
    true_mask[20:80, 10:70] = 1
    true_mask[50:90, 50:110] = 2

    # Calculate and print the loss
    loss_fn = InstanceSegmentationLoss()
    loss_value = loss_fn(pred_mask.long(), true_mask.long())
    print("Loss:", loss_value.item())  # Output will be close to zero for perfect overlaps

    # Example Usage
    # Define a dummy prediction and ground truth masks
    pred_mask = torch.randn(1, 3, 128, 128)  # Example: 3 classes (including background)
    true_mask = torch.randint(0, 3, (1, 128, 128))
    
    # Calculate and print the loss
    loss_fn = InstanceSegmentationLoss()
    loss_value = loss_fn(pred_mask, true_mask)
    print("Loss:", loss_value.item())
