import torch

from dbgprint import dbgprint
from dbgprint import *

def compute_loss(gt_tensors, pred_tensors, thresholds, weights):
	"""
	Compute a loss function based on the distances and thresholds for bounding box coordinates,
	white pixel counts, widths, heights, diagonals, and roundness indices.

	Args:
		gt_tensors (tuple): Ground truth tensors from `extract_bounding_boxes`.
		pred_tensors (tuple): Predicted tensors from `extract_bounding_boxes`.
		thresholds (dict): Thresholds for each variable. Example:
			{
				'bboxes': 50,
				'white_pixels': 100,
				'widths': 20,
				'heights': 20,
				'diagonals': 30,
				'roundness': 0.1
			}
		weights (dict): Weights for each variable. Example:
			{
				'bboxes': 1.0,
				'white_pixels': 0.5,
				'widths': 0.5,
				'heights': 0.5,
				'diagonals': 0.5,
				'roundness': 0.5
			}

	Returns:
		torch.Tensor: Scalar loss value for the batch.
	"""
	# Unpack tensors for ground truth and predictions
	gt_bboxes, gt_white_pixels, gt_widths, gt_heights, gt_diags, gt_roundness_indices = gt_tensors
	pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness_indices = pred_tensors

	# Initialize total loss
	total_loss = 0.0

	# Compute bounding box loss
	# Bounding boxes are in (x1, y1, x2, y2) format
	bbox_threshold = thresholds['bboxes']
	bbox_weight = weights['bboxes']

	# Compute the distance between corners of bounding boxes
	gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2  # (x_center, y_center) for gt
	pred_centers = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2  # (x_center, y_center) for pred

	bbox_distances = torch.norm(gt_centers - pred_centers, dim=-1)  # Euclidean distance per box
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'bbox_distances = {bbox_distances}')
	bbox_loss = torch.where(
		bbox_distances <= bbox_threshold,
		(bbox_distances / bbox_threshold),  # Linear penalty for distances within the threshold
		((bbox_distances - bbox_threshold) ** 2) / bbox_threshold  # Quadratic penalty for exceeding the threshold
	)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'bbox_loss = {bbox_loss}')
	total_loss += bbox_weight * bbox_loss.sum(dim=-1).mean()  # Aggregate loss for bounding boxes
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Loss after BBoxes: {total_loss}')

	# Compute white pixel loss
	white_pixel_threshold = thresholds['white_pixels']
	white_pixel_weight = weights['white_pixels']

	white_pixel_diff = torch.abs(gt_white_pixels - pred_white_pixels)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'white_pixel_diff = {white_pixel_diff}')
	white_pixel_loss = torch.where(
		white_pixel_diff <= white_pixel_threshold,
		(white_pixel_diff / white_pixel_threshold),
		((white_pixel_diff - white_pixel_threshold) ** 2) / white_pixel_threshold
	)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'white_pixel_loss = {white_pixel_loss}')
	total_loss += white_pixel_weight * white_pixel_loss.mean()
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Loss after White Pixels: {total_loss}')

	# Compute width loss
	width_threshold = thresholds['widths']
	width_weight = weights['widths']

	width_diff = torch.abs(gt_widths - pred_widths)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'width_diff = {width_diff}')
	width_loss = torch.where(
		width_diff <= width_threshold,
		(width_diff / width_threshold),
		((width_diff - width_threshold) ** 2) / width_threshold
	)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'width_loss = {width_loss}')
	total_loss += width_weight * width_loss.mean()
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Loss after Width: {total_loss}')

	# Compute height loss
	height_threshold = thresholds['heights']
	height_weight = weights['heights']

	height_diff = torch.abs(gt_heights - pred_heights)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'height_diff = {height_diff}')
	height_loss = torch.where(
		height_diff <= height_threshold,
		(height_diff / height_threshold),
		((height_diff - height_threshold) ** 2) / height_threshold
	)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'height_loss = {height_loss}')
	total_loss += height_weight * height_loss.mean()
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Loss after Height: {total_loss}')

	# Compute diagonal loss
	diagonal_threshold = thresholds['diagonals']
	diagonal_weight = weights['diagonals']

	diagonal_diff = torch.abs(gt_diags - pred_diags)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'diagonal_diff = {diagonal_diff}')
	diagonal_loss = torch.where(
		diagonal_diff <= diagonal_threshold,
		(diagonal_diff / diagonal_threshold),
		((diagonal_diff - diagonal_threshold) ** 2) / diagonal_threshold
	)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'diagonal_loss = {diagonal_loss}')
	total_loss += diagonal_weight * diagonal_loss.mean()
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Loss after Diagonal: {total_loss}')

	# Compute roundness index loss
	roundness_threshold = thresholds['roundness']
	roundness_weight = weights['roundness']

	roundness_diff = torch.abs(gt_roundness_indices - pred_roundness_indices)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'roundness_diff = {roundness_diff}')
	roundness_loss = torch.where(
		roundness_diff <= roundness_threshold,
		(roundness_diff / roundness_threshold),
		((roundness_diff - roundness_threshold) ** 2) / roundness_threshold
	)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'roundness_loss = {roundness_loss}')
	total_loss += roundness_weight * roundness_loss.mean()
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Loss after Roundness: {total_loss}')

	return total_loss
