#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceSegmentationLoss(nn.Module):
    """
    Computes the loss for instance segmentation given predicted masks and ground truth masks.
    """
    def __init__(self):
        super(InstanceSegmentationLoss, self).__init__()

    def convert_mask_to_binary_masks(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Converts an instance mask (where each unique integer is an instance)
        to a tensor of binary instance masks

        Args:
            mask: The input instance mask

        Returns: A tensor of boolean binary instance masks
        """
        unique_instances = torch.unique(mask)
        binary_masks = []
        for instance_id in unique_instances:
            if instance_id == 0:  # skip background
                continue
            binary_mask = (mask == instance_id)
            binary_masks.append(binary_mask)
        return torch.stack(binary_masks, dim=0)

    def calculate_iou(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Intersection over Union (IoU) between two masks

        Args:
            pred_mask: Predicted boolean tensor mask
            true_mask: Ground Truth boolean tensor mask

        Returns: IoU value
        """
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        union = torch.logical_or(pred_mask, true_mask).sum()
        iou = torch.where(union == 0, torch.tensor(0.0, device=pred_mask.device), intersection / union)
        return iou

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss given predicted masks and ground truth masks

        Args:
          pred_mask: Predicted mask as a PyTorch Tensor (H, W), with different integers for each instance
          true_mask: True mask as a PyTorch Tensor (H, W), with different integers for each instance
        Returns:
          loss: The calculated loss as a PyTorch tensor (scalar)
        """
        # Convert masks to binary masks
        pred_binary_masks = self.convert_mask_to_binary_masks(pred_mask)
        true_binary_masks = self.convert_mask_to_binary_masks(true_mask)

        # Calculate loss for predicted mask vs. closest ground truth
        pred_losses = torch.zeros(pred_binary_masks.shape[0], device=pred_mask.device)
        for i, pred_binary_mask in enumerate(pred_binary_masks):
            ious = torch.zeros(true_binary_masks.shape[0], device=pred_mask.device)
            for j, true_binary_mask in enumerate(true_binary_masks):
                ious[j] = self.calculate_iou(pred_binary_mask, true_binary_mask)
            max_iou_idx = torch.argmax(ious)
            pred_losses[i] = 1 - ious[max_iou_idx]

        # Calculate loss for ground truth masks vs. closest prediction
        true_losses = torch.zeros(true_binary_masks.shape[0], device=pred_mask.device)
        for i, true_binary_mask in enumerate(true_binary_masks):
            ious = torch.zeros(pred_binary_masks.shape[0], device=pred_mask.device)
            for j, pred_binary_mask in enumerate(pred_binary_masks):
                ious[j] = self.calculate_iou(pred_binary_mask, true_binary_mask)
            max_iou_idx = torch.argmax(ious)
            true_losses[i] = 1 - ious[max_iou_idx]

        # Calculate average loss
        total_loss = pred_losses.sum() + true_losses.sum()
        num_instances = pred_binary_masks.shape[0] + true_binary_masks.shape[0]
        if num_instances == 0:
            return torch.tensor(0.0, device=pred_mask.device, requires_grad=True)  # Avoid divide by zero
        avg_loss = total_loss / num_instances
        return avg_loss

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
