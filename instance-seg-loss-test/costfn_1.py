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

'''
def get_bounding_boxes(mask_tensor):
  """
  Computes bounding boxes for binary segmentation masks.

  Args:
    mask_tensor: A PyTorch tensor of shape [B, N, H, W] representing a batch of binary segmentation masks.

  Returns:
    A tensor of shape [B, N, 4] representing the bounding boxes, 
    where each box is in the format [x_min, y_min, x_max, y_max].
  """

  mask_nonzero = torch.nonzero(mask_tensor, as_tuple=False).T
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.min(mask_nonzero, dim=0)[:2][0] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.min(mask_nonzero, dim=0)[:2][1] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.max(mask_nonzero, dim=0)[:2][0] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.max(mask_nonzero, dim=0)[:2][1] = }')

  # Get the coordinates of the bounding boxes
  min_vals = torch.min(mask_nonzero, dim=0)[:2].unsqueeze(1)
  max_vals = torch.max(mask_nonzero, dim=0)[:2].unsqueeze(1)

  # Combine min and max values to form bounding boxes
  bounding_boxes = torch.cat((min_vals, max_vals), dim=1)
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes.shape = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes = }')


  # Find the coordinates of non-zero elements
  nonzero_indices = torch.nonzero(mask_tensor)
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{nonzero_indices.shape = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'nonzero_indices: {nonzero_indices}')
  for i in range(len(nonzero_indices)):
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{nonzero_indices[i] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{nonzero_indices[:, 0] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{nonzero_indices[:, 1] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{nonzero_indices[:, 2] = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{nonzero_indices[:, 3] = }')


  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.min(nonzero_indices[:, 2], dim=0) = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.min(nonzero_indices[:, 3], dim=0) = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.max(nonzero_indices[:, 2], dim=0) = }')
  dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.max(nonzero_indices[:, 3], dim=0) = }')

  # Get the minimum and maximum coordinates for each mask
  x_min = torch.min(nonzero_indices[:, 2], dim=0)[0]
  y_min = torch.min(nonzero_indices[:, 3], dim=0)[0]
  x_max = torch.max(nonzero_indices[:, 2], dim=0)[0]
  y_max = torch.max(nonzero_indices[:, 3], dim=0)[0]

  # Stack the coordinates to form the bounding boxes
  bounding_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)

  return bounding_boxes
'''

def extract_bounding_boxes(masks):
    """
    Extract bounding boxes, white pixel count, width, height, diagonal, and roundness index from binary masks.
    
    Args:
        masks (torch.Tensor): 4D tensor of binary masks (BxCxHxW).
    
    Returns:
        tuple: A tuple containing the following tensors:
            - bounding_boxes (torch.Tensor): Bounding box coordinates (Bx4), format: (x1, y1, x2, y2).
            - white_pixels (torch.Tensor): Number of white pixels in each mask (B).
            - widths (torch.Tensor): Width of each bounding box (B).
            - heights (torch.Tensor): Height of each bounding box (B).
            - diagonals (torch.Tensor): Diagonal length of each bounding box (B).
            - roundness_indices (torch.Tensor): Roundness index of the white content in each mask (B).
    """
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{masks.shape = }')

    max_dbg_idx = min(masks.shape[1]-1, 5)

    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{masks[0, max_dbg_idx].shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{masks[0, max_dbg_idx] = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.nonzero(masks[0, max_dbg_idx].view(-1), as_tuple=False) = }')
    for idx in range(max_dbg_idx):
        dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, idx]) = }')
    #dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, 1]) = }')
    #dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, 2]) = }')
    #dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, 3]) = }')
    #dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, 4]) = }')
    #dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0, 5]) = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{torch.sum(masks[0], dim=(1, 2)) = }')
    B, N, H, W = masks.shape

    '''    
    # This one takes one order magnitude more than the other...
    start = datetime.datetime.now()
    bounding_boxes = masks_to_bounding_boxes_vectorized(masks)
    end   = datetime.datetime.now()
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'elapsed time: {end-start}')

    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes = }')
    '''    

    start = datetime.now()
    bounding_boxes = get_bounding_boxes(masks)
    end   = datetime.now()
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'elapsed time: {end-start}')

    dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'{bounding_boxes.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'{bounding_boxes[0:max_dbg_idx] = }')
    #white_pixels, widths, heights, diagonals, roundness_indices = None, None, None, None, None
    #return bounding_boxes, white_pixels, widths, heights, diagonals, roundness_indices


    white_pixels = num_white_pixels_in_binary_mask(masks)
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{white_pixels.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{white_pixels = }')

    '''
    # Find the coordinates of white pixels
    #white_coords = torch.nonzero(masks.view(B, N, -1), as_tuple=False)
    white_coords = torch.sum(masks.view(B, N, -1), dim=2)
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{white_coords.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{white_coords[0] = }')
    
    # Extract bounding box coordinates
    min_coords, _ = white_coords.view(B, N, -1, 2).min(dim=2)
    max_coords, _ = white_coords.view(B, N, -1, 2).max(dim=2)
    bounding_boxes = torch.cat([min_coords, max_coords], dim=1)
    dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'{bounding_boxes.shape = }')
    '''

    '''
            y_min = white_pixels[0].min()
            x_min = white_pixels[1].min()
            y_max = white_pixels[0].max()
            x_max = white_pixels[1].max()

            bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
    '''
    
    '''
    # Count the number of white pixels in each mask
    white_pixels = masks.view(B, -1).sum(dim=1)
    '''
    
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
    roundness_indices = 4 * torch.pi * areas / (perimeters**2 + 1e-6)
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{roundness_indices.shape = }')
    dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{roundness_indices = }')
    
    return bounding_boxes, white_pixels, widths, heights, diagonals, roundness_indices


def instance_segmentation_cost(bounding_boxes, white_pixels, widths, heights, diagonals, roundness_indices,
                               bbox_weight=10.0, wh_weight=5.0, pixel_weight=3.0, diag_weight=1.0, round_weight=1.0):
    """
    Calculate the cost for instance segmentation based on extracted features.
    
    Args:
        bounding_boxes (torch.Tensor): Bounding box coordinates (Bx4), format: (x1, y1, x2, y2).
        white_pixels (torch.Tensor): Number of white pixels in each mask (B).
        widths (torch.Tensor): Width of each bounding box (B).
        heights (torch.Tensor): Height of each bounding box (B).
        diagonals (torch.Tensor): Diagonal length of each bounding box (B).
        roundness_indices (torch.Tensor): Roundness index of the white content in each mask (B).
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
    round_cost = round_weight * (1 - roundness_indices)
    
    total_cost = bbox_cost + wh_cost + pixel_cost + diag_cost + round_cost
    return total_cost
