#!/usr/bin/env python3

import sys
import torch
#import datetime

from dbgprint import dbgprint
from dbgprint import *

def masks_to_bounding_boxes_vectorized(masks):
    B, N, H, W = masks.shape
    masks = masks.view(-1, H, W)  # [B*N, H, W]
    non_zero_indices = torch.nonzero(masks, as_tuple=False)
    
    # Initialize bounding box tensor
    bounding_boxes = torch.zeros((B*N, 4), device=masks.device)
    
    unique_batch_mask_ids = torch.unique(non_zero_indices[:, 0])
    for idx in unique_batch_mask_ids:
        mask_indices = non_zero_indices[non_zero_indices[:, 0] == idx]
        x_coords, y_coords = mask_indices[:, 2], mask_indices[:, 1]
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = torch.min(x_coords), torch.max(x_coords)
            y_min, y_max = torch.min(y_coords), torch.max(y_coords)
            bounding_boxes[idx, :] = torch.tensor([x_min, y_min, x_max, y_max])
    
    # Reshape back to [B, N, 4]
    bounding_boxes = bounding_boxes.view(B, N, 4)
    
    return bounding_boxes

def num_white_pixels_in_binary_mask(masks):
	"""
	Returns the number of white pixels in each binary mask
	
	Args:
		masks: torch.Tensor of shape [B,N,H,W] with binary values (0 or 1)
			  B: batch size, N: number of masks per batch, H: height, W: width
	
	Returns:
		num of white pixels: torch.Tensor of shape [B,N]
		e.g. tensor([8047., 5451., 5309., 4592., 3030., 2217., 1333., 1272., 1219., 1128., 1112., 1086., 1053.], device='cuda:1')
	"""
	white_px = torch.sum(masks, dim=(2, 3))
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{white_px = }')
	return white_px

def get_bounding_boxes(masks):
    """
    Extract bounding boxes from binary segmentation masks.
    
    Args:
        masks: torch.Tensor of shape [B,N,H,W] with binary values (0 or 1)
              B: batch size, N: number of masks per batch, H: height, W: width
    
    Returns:
        boxes: torch.Tensor of shape [B,N,4] containing [x_min, y_min, x_max, y_max] for each mask
    """
    B, N, H, W = masks.shape
    
    # Get indices of all non-zero elements
    y_coords = torch.arange(H, device=masks.device).view(1, 1, H, 1).expand(B, N, H, W)
    x_coords = torch.arange(W, device=masks.device).view(1, 1, 1, W).expand(B, N, H, W)
    
    # Mask the coordinates with the binary mask
    y_coords = y_coords * masks
    x_coords = x_coords * masks
    
    # For each mask, get min and max coordinates
    # Replace 0s with a large number for min operation and small number for max operation
    y_coords_min = y_coords.masked_fill(masks == 0, float('inf'))
    y_coords_max = y_coords.masked_fill(masks == 0, float('-inf'))
    x_coords_min = x_coords.masked_fill(masks == 0, float('inf'))
    x_coords_max = x_coords.masked_fill(masks == 0, float('-inf'))
    
    # Get min and max values for x and y coordinates
    y_min = y_coords_min.amin(dim=(2, 3))  # [B,N]
    y_max = y_coords_max.amax(dim=(2, 3))  # [B,N]
    x_min = x_coords_min.amin(dim=(2, 3))  # [B,N]
    x_max = x_coords_max.amax(dim=(2, 3))  # [B,N]
    
    # Stack the coordinates to get final bounding boxes
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=2)  # [B,N,4]
    
    # Handle empty masks (all zeros)
    empty_masks = (masks.sum(dim=(2,3)) == 0)  # [B,N]
    boxes[empty_masks] = 0
    
    return boxes

def extract_bounding_boxes(masks):
    """
    Extract bounding boxes, white pixel count, width, height, diagonal, and roundness index from binary masks.
    
    Args:
        masks (torch.Tensor): 4D tensor of binary masks (BxNxHxW).
    
    Returns:
        tuple: A tuple containing the following tensors:
            - bounding_boxes (torch.Tensor): Bounding box coordinates (BxNx4), format: (x1, y1, x2, y2).
            - white_pixels (torch.Tensor): Number of white pixels in each mask (BxN).
            - widths (torch.Tensor): Width of each bounding box (BxN).
            - heights (torch.Tensor): Height of each bounding box (BxN).
            - diagonals (torch.Tensor): Diagonal length of each bounding box (BxN).
            - roundness (torch.Tensor): Roundness index of the white content in each mask (BxN).
    """
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{masks.shape = }')

    max_dbg_idx = min(masks.shape[1]-1, 5)

    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{masks[0, max_dbg_idx].shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{masks[0, max_dbg_idx] = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.nonzero(masks[0, max_dbg_idx].view(-1), as_tuple=False) = }')
    for idx in range(max_dbg_idx):
        dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, idx]) = }')

    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0], dim=(1, 2)) = }')
    B, N, H, W = masks.shape

    start = datetime.now()
    bounding_boxes = get_bounding_boxes(masks)
    end   = datetime.now()
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'elapsed time: {end-start}')

    dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'{bounding_boxes.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'{bounding_boxes[0:max_dbg_idx] = }')


    white_pixels = num_white_pixels_in_binary_mask(masks)
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{white_pixels.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{white_pixels = }')

    # Calculate width, height, and diagonal of each bounding box
    widths	= bounding_boxes[:, :, 2] - bounding_boxes[:, :, 0] + 1
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{widths.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{widths = }')
    heights	= bounding_boxes[:, :, 3] - bounding_boxes[:, :, 1] + 1
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{heights.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{heights = }')
    diagonals	= torch.sqrt(widths**2 + heights**2)
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{diagonals.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{diagonals = }')
    
    # Calculate roundness index
    areas = white_pixels
    perimeters = 2 * (widths + heights)
    roundness = 4 * torch.pi * areas / (perimeters**2 + 1e-6)
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{roundness.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{roundness = }')
    
    return bounding_boxes, white_pixels, widths, heights, diagonals, roundness


def instance_segmentation_cost(bounding_boxes, white_pixels, widths, heights, diagonals, roundness,
                               bbox_weight=10.0, wh_weight=5.0, pixel_weight=3.0, diag_weight=1.0, round_weight=1.0):
    """
    Calculate the cost for instance segmentation based on extracted features.
    
    Args:
        bounding_boxes (torch.Tensor): Bounding box coordinates (Bx4), format: (x1, y1, x2, y2).
        white_pixels (torch.Tensor): Number of white pixels in each mask (B).
        widths (torch.Tensor): Width of each bounding box (B).
        heights (torch.Tensor): Height of each bounding box (B).
        diagonals (torch.Tensor): Diagonal length of each bounding box (B).
        roundness (torch.Tensor): Roundness index of the white content in each mask (B).
        bbox_weight (float): Weight for bounding box position cost.
        wh_weight (float): Weight for width/height cost.
        pixel_weight (float): Weight for white pixel count cost.
        diag_weight (float): Weight for diagonal length cost.
        round_weight (float): Weight for roundness index cost.
    
    Returns:
        torch.Tensor: Instance segmentation cost (B).
    """
    bbox_cost = bbox_weight * (bounding_boxes[:, 2] - bounding_boxes[:, 0] + bounding_boxes[:, 3] - bounding_boxes[:, 1])
    wh_cost = wh_weight * (widths + heights)
    pixel_cost = pixel_weight * white_pixels
    diag_cost = diag_weight * diagonals
    round_cost = round_weight * (1 - roundness)
    
    total_cost = bbox_cost + wh_cost + pixel_cost + diag_cost + round_cost
    return total_cost
