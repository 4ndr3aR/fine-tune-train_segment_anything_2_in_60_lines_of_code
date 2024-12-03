#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Tuple

class InstanceSegmentationLoss(nn.Module):
    def __init__(self):
        super(InstanceSegmentationLoss, self).__init__()

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss given predicted masks and ground truth masks

        Args:
          pred_mask: Predicted mask as a PyTorch Tensor (H, W), with different integers for each instance
          true_mask: True mask as a PyTorch Tensor (H, W), with different integers for each instance
        Returns:
          loss: The calculated loss as a PyTorch tensor (scalar)
        """

        pred_mask = pred_mask.detach().cpu()
        true_mask = true_mask.detach().cpu()

        # Get unique instance IDs from both masks
        unique_pred_instances = torch.unique(pred_mask)
        unique_true_instances = torch.unique(true_mask)

        # Skip background (instance ID 0)
        unique_pred_instances = unique_pred_instances[unique_pred_instances != 0]
        unique_true_instances = unique_true_instances[unique_true_instances != 0]

        total_loss = 0.0
        num_instances = 0.0

        # Calculate loss for predicted mask vs. closest ground truth
        for pred_instance_id in unique_pred_instances:
            max_iou = 0
            pred_binary_mask = (pred_mask == pred_instance_id).float()
            for true_instance_id in unique_true_instances:
                true_binary_mask = (true_mask == true_instance_id).float()
                iou = self.calculate_iou(pred_binary_mask, true_binary_mask)
                max_iou = max(max_iou, iou)
            total_loss += 1 - max_iou
            num_instances += 1

        # Calculate loss for ground truth masks vs. closest prediction
        for true_instance_id in unique_true_instances:
            max_iou = 0
            true_binary_mask = (true_mask == true_instance_id).float()
            for pred_instance_id in unique_pred_instances:
                pred_binary_mask = (pred_mask == pred_instance_id).float()
                iou = self.calculate_iou(pred_binary_mask, true_binary_mask)
                max_iou = max(max_iou, iou)
            total_loss += 1 - max_iou
            num_instances += 1

        if num_instances == 0:
            return torch.tensor(0.0, requires_grad=True)  # Avoid divide by zero

        avg_loss = total_loss / num_instances
        return torch.tensor(avg_loss, requires_grad=True)

    def calculate_iou(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
        """
        Calculates the Intersection over Union (IoU) between two masks
        Args:
            pred_mask: Predicted boolean PyTorch Tensor mask
            true_mask: Ground Truth boolean PyTorch Tensor mask
        Returns: IoU value
        """
        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()
        if union == 0:
            return 0
        return intersection / union

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
    loss_value = loss_fn(pred_mask, true_mask)
    print("Loss:", loss_value.item())  # Output will be close to zero for perfect overlaps
