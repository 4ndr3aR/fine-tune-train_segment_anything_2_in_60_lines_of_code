import os

import cv2

import random

import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from dbgprint import dbgprint
from dbgprint import *

from instance_seg_loss import read_color_palette, read_colorid_file

from utils import draw_points_on_image

def get_random_tree_mask(seg_mask, small_mask):
  """
  Extracts a random tree instance mask from instance segmentation data,
  based on the given trunk segmentation mask.

  Args:
    seg_mask: A binary mask (480x270) where 1 represents tree trunks.
    small_mask: An RGB instance segmentation mask (480x270x3) with unique
               colors for each instance.

  Returns:
    A tuple containing:
      - tree_mask: A binary mask (480x270) representing a single tree
        instance.
      - random_coord: A tuple (row, col) representing the coordinates of the 
        randomly selected point on a tree trunk.
  """
  dbgprint(dataloader, LogLevel.INFO, f'{type(seg_mask) = } - {seg_mask.shape = } - {seg_mask.dtype = }')
  dbgprint(dataloader, LogLevel.INFO, f'{type(small_mask) = } - {small_mask.shape = } - {small_mask.dtype = }')
  # 1. Find coordinates of all tree trunks
  trunk_coords = np.argwhere(seg_mask == 1)
  dbgprint(dataloader, LogLevel.INFO, f'trunk_coords = {trunk_coords}')

  if len(trunk_coords) == 0:
    return np.zeros_like(seg_mask, dtype=np.uint8), None,  # No trunks found, return empty mask and None coord
    
  # 2. Randomly select a single trunk coordinate
  #random_coord = tuple(random.choice(trunk_coords))
  random_trunk_coord = np.random.choice(len(trunk_coords), size=1)[0]
  random_coord = trunk_coords[random_trunk_coord]
  dbgprint(dataloader, LogLevel.INFO, f'random_coord = {random_coord}')

  # 3. Get the RGB color at that coordinate from small_mask
  selected_color = small_mask[random_coord[0], random_coord[1]]

  # 4. Create a binary mask for the selected tree
  tree_mask = np.all(small_mask == selected_color, axis=-1).astype(np.uint8)
  
  return tree_mask, random_coord #[random_coord[1], random_coord[0]]


def select_random_tree(seg_mask, small_mask):
    """
    Selects a random tree from an instance segmentation mask and returns a binary mask for that tree.

    Parameters:
    - seg_mask: np.array of shape [480, 270] where 0 is background and 1 represents tree trunks.
    - small_mask: np.array of shape [480, 270, 3] where each object has a unique color.

    Returns:
    - tree_mask: np.array of shape [480, 270] where 1 indicates the selected tree and 0 is everything else.
    - coordinate: Tuple (y, x) representing a random point within the selected tree trunk.
    """
    
    # Find all coordinates where seg_mask is 1 (tree trunks)
    trunk_coords = np.argwhere(seg_mask == 1)
    dbgprint(dataloader, LogLevel.TRACE, f'trunk_coords = {trunk_coords}')
    
    if len(trunk_coords) == 0:
        return np.zeros_like(seg_mask, dtype=np.uint8), None,  # No trunks found, return empty mask and None coord
    #if not trunk_coords.size:
    #    raise ValueError("No tree trunks found in the segmentation mask.")
    
    # Choose a random coordinate from the trunk coordinates
    x, y = trunk_coords[random.randint(0, len(trunk_coords) - 1)]
    dbgprint(dataloader, LogLevel.TRACE, f'random_coord = {(x, y)}')
    
    # Get the color of the tree at this coordinate from small_mask
    tree_color = tuple(small_mask[x, y])
    
    # Create a mask where only pixels of this color are set to 1
    tree_mask = np.all(small_mask == tree_color, axis=-1).astype(np.uint8)
   
    return tree_mask, (x, y)



def find_valid_centroid(mask):
    # Calculate image moments
    M = cv2.moments(mask)

    # Calculate the raw centroid coordinates
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # Fallback if no white pixel is detected
        return None

    # If the centroid is already within the mask, return it
    if mask[cY, cX] > 0:
        return (cX, cY)

    # Otherwise, use the contours to find the nearest point inside the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)		# same
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)	# same again

    # Find the point on the contour closest to the raw centroid
    min_dist = float("inf")
    closest_point = (cX, cY)

    for contour in contours:
        for point in contour:
            px, py = point[0]
            dist = np.sqrt((px - cX) ** 2 + (py - cY) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_point = (px, py)

    return closest_point


def get_bbox(mask):
	x, y, w, h = cv2.boundingRect(mask)
	bbox = (x, y, x + w, y + h)
	return bbox

def get_all_trees(seg_mask, iseg_mask, px_threshold=-1, px_threshold_perc=-1, trunk_to_leaves_ratio=0.05, debug_save_all_masks=False):
	"""
	Extracts all tree instance masks, their center points, and bounding boxes.

	Args:
		seg_mask: A binary mask (480x270) where 1 represents tree trunks.
		iseg_mask: An RGB instance segmentation mask (480x270x3) with unique
			colors for each tree instance.

	Returns:
		A list of tuples, where each tuple contains:
			- tree_mask: A binary mask (480x270) representing a single tree instance.
			- center_point: A tuple (row, col) representing the center of the tree mask.
			- bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
			- color: the color of the instance segmentation mask returned
	"""
	# 1. Mask out background in iseg_mask using seg_mask
	masked_iseg = iseg_mask * np.expand_dims(seg_mask, axis=-1)
	
	# 2. Get unique colors (representing different trees) from the masked segmentation
	unique_colors = np.unique(masked_iseg.reshape(-1, 3), axis=0)
	
	# Remove background (black) color [0, 0, 0]
	#unique_colors = unique_colors[~np.all(unique_colors == 0, axis=1)]

	trunk_coords = np.argwhere(seg_mask == 1)

	results = []
	subj_color = [6, 108, 153]		# this is the "main subject" color, we'll make an exception for it

	# 3. Create binary mask for each color (tree instance)
	for color in unique_colors:
		if list(color) == [0, 0, 0] or list(color) == [255, 255, 255]:
			continue
		# grab all the pixels corresponding to the current color (e.g. usually a single tree)
		tree_mask	= np.all(iseg_mask == color, axis=-1).astype(np.uint8)
		dbgprint(dataloader, LogLevel.TRACE, f'{seg_mask.shape = } - {tree_mask.shape = } - {seg_mask.dtype = } - {tree_mask.dtype = }')
		# mask "this tree" (dirty as hell usually) with all the seg. map ("all" the trunks)
		tree_trunk	= seg_mask & tree_mask
		# count the pixels in the current single tree mask
		nonzero		= np.count_nonzero(tree_mask)
		rgb_tree_mask	= cv2.bitwise_and(iseg_mask, iseg_mask, mask=tree_mask)
		if debug_save_all_masks:
			#rgb_tree_mask	= cv2.circle(rgb_tree_mask, (int(x), int(y)), radius, color, thickness)
			cv2.imwrite(f'/tmp/instance-seg-mask-{nonzero}px-{color[0]:02X}{color[1]:02X}{color[2]:02X}.png', rgb_tree_mask)
			cv2.imwrite(f'/tmp/tree-trunk-{nonzero}px-{color[0]:02X}{color[1]:02X}{color[2]:02X}.png', tree_trunk)
		dbgprint(dataloader, LogLevel.DEBUG, f'get_all_trees() - Considering color: {color} - nonzero = {nonzero} - px_threshold = {px_threshold} - px_threshold_perc = {px_threshold_perc}')
		if nonzero < px_threshold or nonzero < px_threshold_perc * iseg_mask.shape[0] * iseg_mask.shape[1] / 100.0:
			dbgprint(dataloader, LogLevel.TRACE, f'get_all_trees() - nonzero = {nonzero} - px_threshold = {px_threshold} - px_threshold_perc = {px_threshold_perc * iseg_mask.shape[0] * iseg_mask.shape[1] / 100.0} - DISCARDING TREE - condition 1')
			continue
		nonzero_tree_trunk = np.count_nonzero(tree_trunk)			# count the pixels in the current single tree mask, masked by all the trunks (logical AND)
		dbgprint(dataloader, LogLevel.TRACE, f'get_all_trees() - nonzero_tree_trunk = {nonzero_tree_trunk}')
		if list(color) != subj_color:						# always keep that brown-beige color that is the main subject of the scene, whatever it takes
			if nonzero_tree_trunk < px_threshold or nonzero_tree_trunk < px_threshold_perc * iseg_mask.shape[0] * iseg_mask.shape[1] / 100.0:
				# here we basically throw away everything that has a trunk too small (because it's usually "visual noise", very little info in there)
				dbgprint(dataloader, LogLevel.TRACE, f'get_all_trees() - tree trunk mask = {nonzero_tree_trunk} - px_threshold = {px_threshold} - px_threshold_perc = {px_threshold_perc * iseg_mask.shape[0] * iseg_mask.shape[1] / 100.0} - DISCARDING TREE - condition 2')
				continue
			if nonzero_tree_trunk < trunk_to_leaves_ratio * nonzero:	# if the "leftover trunk mask" is less than 5% of the original single tree mask, discard
				# usually this means it's not a tree... it's something else, usually background, grass or whatever. Sometimes it's a tree but for any reasons,
				# it has no trunk (e.g. it's a bunch of large leaves in the foreground)
				dbgprint(dataloader, LogLevel.TRACE, f'get_all_trees() - tree trunk mask = {nonzero_tree_trunk} px - tree mask = {nonzero} px - DISCARDING TREE - condition 3')
				continue
	
		'''
		# 3b. clean the masks from all those random pixels and blobs that are too small for being meaningful/useful
		tree_mask = extract_blobs_above_threshold(tree_mask, threshold=5,
								cluster_centroid_distance=50,
								distant_blobs_threshold_px=50,
								large_blob_threshold=100)
		'''

		blob_min_threshold = 5
		largest_blob, all_blobs = extract_blobs_above_threshold(tree_mask, threshold=blob_min_threshold,
									cluster_centroid_distance=50,
									distant_blobs_threshold_px=75,
									large_blob_threshold=100)

		for label_id, area in enumerate(all_blobs):
			dbgprint(dataloader, LogLevel.DEBUG, f'Mask with color {color} - has blob {label_id} with area {area} - threshold {blob_min_threshold}')

		tree_mask = largest_blob & tree_mask
	
		# 4. Calculate bounding box using OpenCV
		bbox = get_bbox(tree_mask)

		# 5. Calculate tree center (centroid) from the mask
		coords = np.column_stack(np.where(tree_mask == 1))
		dbgprint(dataloader, LogLevel.TRACE,		f'get_all_trees() - Found tree coordinates: {len(coords)} - {coords}')

		if len(coords) > 0:
			#center = coords.mean(axis=0).astype(int)
			center = find_valid_centroid(tree_trunk)
			rgb_tree_mask	= cv2.circle(rgb_tree_mask, (int(center[1]), int(center[0])), 1, (0, 0, 255), -1)
			cv2.imwrite(f'/tmp/instance-seg-mask-{nonzero}px-{color[0]:02X}{color[1]:02X}{color[2]:02X}.png', rgb_tree_mask)
			center_point = (center[1], center[0])  # (row, col)
			dbgprint(dataloader, LogLevel.DEBUG,	f'get_all_trees() - Found center point: {center_point}')
		else:
			center_point = None
			dbgprint(dataloader, LogLevel.WARNING,	f'get_all_trees() - No coordinates found for tree with color: {color}')
			continue

		if seg_mask[center_point[0], center_point[1]] == 0:
			coords_str1 = " ".join([f"({x}, {y})" for x, y in trunk_coords[:3]])
			coords_str2 = " ".join([f"({x}, {y})" for x, y in trunk_coords[-3:]])
			dbgprint(dataloader, LogLevel.WARNING,	f'get_all_trees() - center_point not in trunk_coords: {center_point} - {coords_str1} ... {coords_str2}')
			continue

		# 6. Store result
		results.append((tree_mask, center_point, bbox, color, nonzero, tree_trunk, largest_blob))

	dbgprint(dataloader, LogLevel.DEBUG, f'get_all_trees() - returned {len(results)} trees above the provided pixel threshold')
	return results


def extract_main_blob(image):
	# Ensure the input is a numpy array
	image = np.array(image)

	img = cv2.dilate(image, np.ones((3, 3), np.uint8))

	# Check if the image is 2D
	if img.ndim != 2:
		raise ValueError("Input image must be a 2D array")

	# Preserve original data type and determine the maximum value
	original_dtype = img.dtype
	max_val = img.max() if img.size > 0 else 0

	# Convert the image to uint8 for OpenCV processing
	if img.dtype == bool:
		img_uint8 = img.astype(np.uint8) * 255
	else:
		img_uint8 = img.astype(np.uint8)

	# Perform connected component analysis
	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_uint8)

	# If there are no foreground components, return a blank image
	if num_labels <= 1:
		return np.zeros_like(img, dtype=original_dtype)

	# Identify the largest component (excluding the background)
	areas = stats[1:, cv2.CC_STAT_AREA]
	if len(areas) == 0:
		return np.zeros_like(img, dtype=original_dtype)
	max_area_idx = np.argmax(areas)
	max_label = max_area_idx + 1  # Adjust for background label 0

	# Create the mask for the largest component
	mask = (labels == max_label)

	# Convert the mask to the original data type and scale
	if original_dtype == bool:
		result = mask.astype(bool)
	else:
		result = mask.astype(original_dtype) * max_val

	return result






def extract_blobs_above_threshold(image, 
                                  threshold=5, 
                                  cluster_centroid_distance=50, 
                                  distant_blobs_threshold_px=75, 
                                  large_blob_threshold=75):
	"""
	Written by: o1-2024-12-17

	Returns a mask containing all blobs (connected components) that satisfy:
	  1) blob area >= 'threshold', AND
	  2) either blob is within 'cluster_centroid_distance' of the main cluster 
		 (the largest blob by area), OR
	  3) blob area >= 'large_blob_threshold' (kept regardless of distance).
	  
	Additionally, if a blob is further than 'cluster_centroid_distance' from
	the main cluster centroid and has area < 'distant_blobs_threshold_px',
	it is discarded.

	Parameters
	----------
	image : array_like
		2D input binary (or integer) image.
	threshold : int, optional
		Minimum blob area, in pixels, to even be considered.
	cluster_centroid_distance : float, optional
		Distance threshold from the main cluster's centroid, beyond which small
		blobs are discarded.
	distant_blobs_threshold_px : int, optional
		If a blob's area is below this value, and it lies farther than 
		'cluster_centroid_distance' from the main cluster centroid, discard it.
	large_blob_threshold : int, optional
		If a blob's area is at least this large, keep it regardless of distance.
		
	Returns
	-------
	result : ndarray
		2D binary mask (same shape as 'image') with the kept blobs. Has the same 
		dtype as the original image (bool or integer).
	"""
	# Ensure input is a numpy array
	image = np.array(image)

	image = cv2.dilate(image, np.ones((3, 3), np.uint8))
	image = cv2.erode (image, np.ones((3, 3), np.uint8))

	# Check if image is 2D
	if image.ndim != 2:
		raise ValueError("Input image must be a 2D array")

	# Preserve original data type and determine max value
	original_dtype = image.dtype
	max_val = image.max() if image.size > 0 else 0

	# Convert to uint8 format for OpenCV processing
	if image.dtype == bool:
		image_uint8 = image.astype(np.uint8) * 255
	else:
		image_uint8 = image.astype(np.uint8)

	# Perform connected component analysis
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_uint8)

	# Handle case of no foreground components
	if num_labels <= 1:
		return np.zeros_like(image, dtype=original_dtype), []

	# Find the label with the largest area (excluding background = label 0)
	largest_label = None
	largest_area = 0
	for label_id in range(1, num_labels):
		area = stats[label_id, cv2.CC_STAT_AREA]
		if area > largest_area:
			largest_area = area
			largest_label = label_id

	# If even the largest label doesn't meet the basic threshold, return empty
	if largest_area < threshold:
		return np.zeros_like(image, dtype=original_dtype), []

	# Get centroid of the largest cluster
	main_centroid = centroids[largest_label]  # [y_center, x_center]

	# Decide which labels to keep
	kept_labels = []
	all_blobs   = []
	for label_id in range(1, num_labels):
		area = stats[label_id, cv2.CC_STAT_AREA]

		# Only consider blobs that meet the basic pixel threshold
		dbgprint(dataloader, LogLevel.TRACE, f'Considering blob {label_id} with area {area} and threshold {threshold}')
		all_blobs.append(area)
		if area < threshold:
			dbgprint(dataloader, LogLevel.DEBUG, f'Considering blob {label_id} with area {area} and threshold {threshold} - DISCARDED - reason 1 - too small to be even considered')
			continue

		# Always keep the largest cluster
		if label_id == largest_label:
			kept_labels.append(label_id)
			continue

		# Compute distance to the main cluster's centroid
		blob_centroid = centroids[label_id]
		dist = np.linalg.norm(blob_centroid - main_centroid)

		# Check distance criteria
		if dist > cluster_centroid_distance and area < distant_blobs_threshold_px:
			# Far away AND too small -> discard
			dbgprint(dataloader, LogLevel.DEBUG, f'Considering blob {label_id} with area {area} and threshold {threshold} - DISCARDED - reason 2 - far away (dist: {dist} px > {cluster_centroid_distance}) and too small (area: {area} < {distant_blobs_threshold_px} px)')
			continue

		# If it is a large blob, keep it regardless of distance
		if area >= large_blob_threshold:
			kept_labels.append(label_id)
			continue

		# Otherwise, if it's within distance or big enough, we keep it
		if dist <= cluster_centroid_distance or area >= distant_blobs_threshold_px:
			kept_labels.append(label_id)

	# Return blank mask if no components are kept
	if not kept_labels:
		return np.zeros_like(image, dtype=original_dtype), []

	# Create combined mask of all kept components
	mask = np.isin(labels, kept_labels)

	# Convert to original data type and scaling
	if original_dtype == bool:
		result = mask.astype(bool)
	else:
		result = mask.astype(original_dtype) * max_val

	return result, all_blobs



def show_instances(image_fn, img, small_mask, seg_mask, all_the_trees):
	masked_iseg   = small_mask * np.expand_dims(seg_mask, axis=-1)
	
	tree_centers  = []
	cv2.destroyAllWindows()
	
	dbgprint(dataloader, LogLevel.INFO, f'Processing image: {image_fn} with {len(all_the_trees)} trees')
	for idx, (tree_mask, center_point, bbox, color, nonzero, tree_trunk, largest_blob) in enumerate(all_the_trees):	# where bbox = (x, y, w, h)
	
		bbox = get_bbox(tree_mask)
		x, y, w, h = bbox
	
		tree_centers.append(center_point)
		dbgprint(dataloader, LogLevel.DEBUG, f'get_all_trees[{idx}] - tree mask shape: {tree_mask.shape} - center point: {center_point} - bbox: {bbox} - color: {color} - nonzero: {nonzero}')
		img_small = cv2.resize(img, (small_mask.shape[1], small_mask.shape[0]))
		img2      = draw_points_on_image(img_small, [list(reversed(center_point))], color=(0, 0, 255), radius=5)
		img2      = cv2.rectangle(img2, (x, y), (w, h), color=(255, 0, 0), thickness=2)
		colored_tree_mask = cv2.bitwise_and(small_mask, small_mask, mask=tree_mask)
	
		white_bg   = cv2.bitwise_not(np.zeros_like(img_small))
		masked_img = cv2.bitwise_and(img_small,	img_small, mask=tree_mask)
		inv_mask   = cv2.bitwise_not(tree_mask*255)
		masked_bg  = cv2.bitwise_and(white_bg,	white_bg,  mask=inv_mask)
		masked_img = cv2.bitwise_or(masked_img, masked_bg)

		mask_uint8 = tree_mask.astype(np.uint8)
		# Find contours using OpenCV
		contours,_ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#contours  = sorted(contours, key=cv2.contourArea, reverse=True)
		approx = []
		for contour in contours:
			peri = cv2.arcLength(contour, True)
			#approx_contour = cv2.approxPolyDP(contour, epsilon=0.02*peri, closed=True)
			approx_contour = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)
			#print(f'approx_contours: {approx_contours}')
			approx.append(approx_contour)

		'''
		for cnt in contours:
			img_cont = cv2.drawContours(img2, [cnt], -1, color=(255,0,0), thickness=3)	
			cv2.imshow("mask contours (COCO)",		img_cont)
			cv2.waitKey(0)
		'''

		orig_img   = img_small.copy()
		img_cont   = img_small.copy()
		img_approx = img_small.copy()

		# Display the extracted tree mask
		cv2.imshow("image_winid",			img2)
		cv2.imshow("segmentation",			seg_mask   * 255)
		cv2.imshow("instance",				small_mask)
		cv2.imshow("colored_tree mask",			colored_tree_mask)
		cv2.imshow("get_all_trees_winid",		tree_mask  * 255)
		cv2.imshow("single tree trunk",			tree_trunk * 255)
		cv2.imshow("largest blob",			largest_blob * 255)
		cv2.imshow('seg mask & iseg mask',		masked_iseg)
		#cv2.imshow("masked image",			img2 & np.dstack((tree_mask*255, tree_mask*255, tree_mask*255)))
		cv2.imshow("masked image",			masked_img)
		img_cont   = cv2.drawContours(img_cont, contours, -1, color=(0,255,0),   thickness=1)
		img_approx = cv2.drawContours(img_approx, approx, -1, color=(0,255,255), thickness=1)
		cv2.imshow("mask contours (COCO)",		img_cont)
		cv2.imshow("mask contours (APPROX)",		img_approx)
		cv2.setWindowTitle("get_all_trees_winid",	f"get_all_trees[{idx}] of {len(all_the_trees)}")
		cv2.setWindowTitle("image_winid",		f"{Path(image_fn).parent.parent.name}/{Path(image_fn).name}")
		cv2.moveWindow("image_winid",			   0, -30)
		cv2.moveWindow("instance",			   0, 360)
		cv2.moveWindow("segmentation",			   0, 680)
		cv2.moveWindow("colored_tree mask",		 490, -30)
		cv2.moveWindow("get_all_trees_winid",		 490, 360)
		cv2.moveWindow("single tree trunk",		 490, 680)
		cv2.moveWindow("masked image",			 980, -30)
		cv2.moveWindow("largest blob",			 980, 360)
		cv2.moveWindow("seg mask & iseg mask",		 980, 680)
		cv2.moveWindow("mask contours (COCO)",		1470, -30)
		cv2.moveWindow("mask contours (APPROX)",	1470, 360)
		cv2.waitKey(0)
		#cv2.destroyAllWindows()

	return tree_centers





class SpreadDataset(Dataset):
	_data = None					# The whole dataset (filenames, imgs, seg. masks, instance seg. masks)
	def __init__(self, data_dir, width=1024, height=1024, num_samples=1024, split="train",
			color_palette_path='./instance-seg-loss-test/color-palette.xlsx',
			train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
			preload=False):

		dbgprint(dataloader, LogLevel.INFO, f'SpreadDataset() - {data_dir} - {width} - {height} - {num_samples} - {split} - {train_ratio} - {val_ratio} - {test_ratio}')

		self.data_dir				= data_dir
		self.split				= split
		self.width				= width
		self.height				= height
		self.num_samples			= num_samples
		self.preload				= preload
		self.color_palette_path			= color_palette_path
		self.color_palette			= None
		self.color_palette_as_list		= None
		self.debug_instance_segmentation_masks	= True
		self.debug_semantic_segmentation_masks	= False
		self.debug_input_points			= False

		if SpreadDataset._data is None:
			SpreadDataset._data = self._load_data(data_dir)

		self.data				= self._split_data(SpreadDataset._data, split, train_ratio, val_ratio, test_ratio)
		if isinstance(self.data, tuple):
			self.train_data, self.val_data, self.test_data = self.data
			self.data = self.data[0]	# if split == "all", we get a tuple of train, val, test
		self.color_palette			= read_color_palette(color_palette_path, invert_to_bgr=False)
		self.color_palette_as_list		= [list(d.values())[0] for d in self.color_palette]
		dbgprint(dataloader, LogLevel.TRACE, f"Loaded color palette: {self.color_palette}")

	def _load_data(self, data_dir):
		# Collect all data entries
		data = []
		for class_dir in os.listdir(data_dir):
			dbgprint(dataloader, LogLevel.INFO, f'Reading directory: {class_dir}')
			if not os.path.isdir(os.path.join(data_dir, class_dir)):
				dbgprint(dataloader, LogLevel.WARNING, f'Scene directory not found or not accessible: {class_dir}')
				continue

			rgb_dir = os.path.join(data_dir, class_dir, "rgb")
			dbgprint(dataloader, LogLevel.INFO, f'RGB directory: {rgb_dir}')
			if not os.path.isdir(rgb_dir) or not os.path.exists(rgb_dir):
				dbgprint(dataloader, LogLevel.WARNING, f'RGB directory not found or not accessible: {rgb_dir}')
				continue

			instance_dir = os.path.join(data_dir, class_dir, "instance_segmentation")
			dbgprint(dataloader, LogLevel.INFO, f'Instance segmentation directory: {instance_dir}')
			if not os.path.isdir(instance_dir) or not os.path.exists(instance_dir):
				dbgprint(dataloader, LogLevel.WARNING, f'Instance segmentation directory not found or not accessible: {instance_dir}')
				continue

			segmentation_dir = os.path.join(data_dir, class_dir, "semantic_segmentation")
			dbgprint(dataloader, LogLevel.INFO, f'Semantic segmentation directory: {segmentation_dir}')
			if not os.path.isdir(segmentation_dir) or not os.path.exists(segmentation_dir):
				dbgprint(dataloader, LogLevel.WARNING, f'Semantic segmentation directory not found or not accessible: {segmentation_dir}')
				continue

			print(f'Reading {class_dir}...', end='', flush=True)
			counter = 0
			for idx, name in enumerate(os.listdir(rgb_dir)):
				if idx % 1000 == 0:
					print('.', end='', flush=True)
				if name.endswith(".png"):
					im_fn      = os.path.join(rgb_dir,          name)
					imask_fn   = os.path.join(instance_dir,     name)
					smask_fn   = os.path.join(segmentation_dir, name)
					colorid_fn = os.path.join(instance_dir,     name.replace(".png", ".txt"))
					dbgprint(dataloader, LogLevel.TRACE, f'{im_fn} - {imask_fn} - {smask_fn} - {colorid_fn} - {self.preload = }')
					data.append({
						"image_fn"	 : im_fn,
						"instance_fn"	 : imask_fn,
						"segmentation_fn": smask_fn,
						"colorid_fn"	 : colorid_fn,
						#"image"		 : cv2.imread(im_fn) if self.preload else None,
						"image"	 : cv2.imread(im_fn)[..., ::-1] if self.preload else None,	# RGB instead of BGRA plz
						"instance"	 : cv2.imread(imask_fn, cv2.IMREAD_UNCHANGED)[..., ::-1] if self.preload else None, # RGB plz
						#"instance"	 : cv2.imread(imask_fn, cv2.IMREAD_UNCHANGED) if self.preload else None,
						"segmentation"	 : cv2.imread(smask_fn, cv2.IMREAD_UNCHANGED) if self.preload else None,            # Grayscale
						"colorids"	 : read_colorid_file(colorid_fn),
					})
					counter += 1
			print(f'{class_dir} has {counter} samples', flush=True)
		print(' done!', flush=True)

		#dbgprint(dataloader, LogLevel.INFO, f"Total number of entries: {len(self.data)}")
		dbgprint(dataloader, LogLevel.INFO, f"Loaded {len(data)} images")

		return data

	def _split_data(self, data, split, train_ratio, val_ratio, test_ratio):
		# Create splits
		train_data, test_data = train_test_split(data,  test_size=test_ratio, random_state=42)
		train_data, val_data  = train_test_split(train_data, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

		# we already print this later
		dbgprint(dataloader, LogLevel.TRACE, f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
		
		if split == "train":
			return train_data
		elif split == "val":
			return val_data
		elif split == "test":
			return test_data
		elif split == "all":
			return train_data, val_data, test_data		# for "external use only"
		else:
			raise Exception(f"Unknown split: {split}")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		ent	= self.data[idx]
		debug_specific_filename = True
		if debug_specific_filename:
			# Ok, there are still a few bugs that "it would be nice" to fix, but they're quite rare cases.
			# The first TODO is to drop altogether all the masks that are very wide and very short (e.g.
			# "squareness index" is low). The second TODO is to is to look inside them and watch at the
			# distribution of pixels... if there are pixels "all the way from left to right of the image",
			# just drop it, it's some weird background. If there are pixels in just two or three blobs,
			# they may be all trees, so split the mask in two or three or whatever, and return the blobs
			# as separate instances.
			debug_basepath		= '/mnt/raid1/dataset/spread/spread'
			debug_category		= 'suburb-us'
			debug_fn		= 'Tree26_1721909084.png'
			debug_fn		= 'Tree101_1721921788.png'
			debug_category		= 'birch-forest'
			debug_fn		= 'Tree8159_1720508295.png'
			debug_fn		= 'Tree27827_1720529494.png'
			debug_fn		= 'Tree49220_1720578742.png'
			debug_category		= 'plantation'
			debug_fn		= 'Tree789_1721038462.png'
			debug_category		= 'broadleaf-forest'
			debug_fn		= 'Tree5396_1720256863.png'
			debug_category		= 'redwood-forest'
			debug_fn		= 'Tree1658_1720853346.png'
			debug_fn		= 'Tree2880_1720832884.png'
			debug_category		= 'downtown-west'
			debug_fn		= 'Tree115_1721227304.png'
			debug_fn		= 'Tree19_1721266439.png'
			debug_fn		= 'Tree60_1721221577.png'
			debug_fn		= 'Tree34_1721294691.png'		# watch this! 😂
			debug_category		= 'city-park'
			debug_fn		= 'Tree4034_1720940143.png'
			debug_fn		= 'Tree4308_1720945515.png'		# TODO: this one is still hard to explain
			debug_fn		= 'Tree1768_1720965873.png'		# TODO: this one also
			debug_category		= 'rainforest'
			debug_fn		= 'Tree3964_1720766396.png'
			debug_fn		= 'Tree6508_1720707560.png'		# OMG watch also this! 🤣
			debug_fn		= 'Tree6174_1720699065.png'

			ent["image_fn"]		= f'{debug_basepath}/{debug_category}/rgb/{debug_fn}'
			ent["segmentation_fn"]	= f'{debug_basepath}/{debug_category}/semantic_segmentation/{debug_fn}'
			ent["instance_fn"]	= f'{debug_basepath}/{debug_category}/instance_segmentation/{debug_fn}'
		dbgprint(dataloader, LogLevel.TRACE, f'Reading images: {ent["image"]} - {ent["instance"]} - {ent["segmentation"]}')
		#img	= cv2.imread(ent["image"])[..., ::-1]					# Convert BGR to RGB
		img = None
		if ent["image"] is not None:
			img = ent["image"]
		else:
			dbgprint(dataloader, LogLevel.INFO, f'Reading image: {ent["image_fn"]}')
			img = cv2.imread(ent["image_fn"])[..., ::-1]
		dbgprint(dataloader, LogLevel.TRACE, f"------------------- Image shape: {img.shape}")
		if img is None:
			dbgprint(dataloader, LogLevel.ERROR, f'Error reading image: {ent["image"]}')
		#ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_UNCHANGED)			# Read as is
		#imask = cv2.imread(ent["instance"], cv2.IMREAD_GRAYSCALE)			# Read grayscale
		#imask	= cv2.imread(ent["instance"],		cv2.IMREAD_UNCHANGED)		# Read as is
		#small_mask = ent["instance"] if ent["instance"] is not None else cv2.imread(ent["instance_fn"], cv2.IMREAD_UNCHANGED)[..., ::-1]
		small_mask = ent["instance"] if ent["instance"] is not None else cv2.imread(ent["instance_fn"], cv2.IMREAD_UNCHANGED)
		if small_mask is None:
			dbgprint(dataloader, LogLevel.ERROR, f'Error reading instance segmentation mask: {ent["instance"]}')
		dbgprint(dataloader, LogLevel.TRACE, f"------------------- Instance segmentation mask shape: {small_mask.shape}")

		seg_mask = ent["segmentation"] if ent["segmentation"] is not None else cv2.imread(ent["segmentation_fn"], cv2.IMREAD_UNCHANGED)
		#cv2.imshow("Segmentation orig", seg_mask * 255)
		if seg_mask is None:
			dbgprint(dataloader, LogLevel.ERROR, f'Error reading segmentation mask: {ent["segmentation"]}')
		dbgprint(dataloader, LogLevel.TRACE, f"------------------- Segmentation mask shape: {seg_mask.shape}")

		#imask	= replace_white_background_with_black(imask)
		#smask	= cv2.imread(ent["segmentation"],	cv2.IMREAD_GRAYSCALE)		# Read grayscale
		color_ids = ent["colorids"] if ent["colorids"] is not None else read_colorid_file(ent["colorid_fn"])		# this returns a colorid_dict
		if color_ids is None:
			dbgprint(dataloader, LogLevel.ERROR, f'Error reading colorid file: {ent["colorid_fn"]}')
		dbgprint(dataloader, LogLevel.TRACE, f'------------------- Color ids: {color_ids}')

		# Resize images and masks to the same resolution
		img      = cv2.resize(img,        (self.width, self.height))
		dbgprint(dataloader, LogLevel.TRACE, f"------------------- Resized image shape: {img.shape}")
		imask    = cv2.resize(small_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

		# we want segmentation masks to be 480x270 to draw coords on trunks and then pick the color from small_masks (instance segmentation masks)
		seg_mask = cv2.resize(seg_mask,   (small_mask.shape[1], small_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

		print(f'img.shape: {img.shape}, imask.shape: {imask.shape}, seg_mask.shape: {seg_mask.shape}, small_mask.shape: {small_mask.shape}')

		tree_centers  = []
		tree_mask     = None

		# TODO: discover why it doesn't work for this: /mnt/raid1/dataset/spread/spread/plantation/semantic_segmentation/Tree789_1721038462.png
		all_the_trees = get_all_trees(seg_mask, small_mask, px_threshold=50, px_threshold_perc=0.01)	# in the end we also set px_threshold, we want at least 50px masks
		if self.debug_instance_segmentation_masks:
			tree_centers = show_instances(ent["image_fn"], img, small_mask, seg_mask, all_the_trees)
		else:
			idx = random.randint(0, len(all_the_trees) - 1)
			tree_centers = all_the_trees[idx][1]
			tree_mask    = all_the_trees[idx][0]

		if False:
			tree_mask, tree_center = select_random_tree(seg_mask, small_mask)
			if tree_center is None:
				dbgprint(dataloader, LogLevel.WARNING, f'Error getting random tree mask - {ent["image_fn"]}')
				tree_center = [0, 0]
	
			if self.debug_semantic_segmentation_masks:
				tree_mask_1, tree_center_1 = get_random_tree_mask(seg_mask, small_mask)
				#tree_mask_2, tree_center_2 = extract_random_tree_2 (seg_mask, small_mask)
				tree_mask_2, tree_center_2 = select_random_tree(seg_mask, small_mask)
	
				if tree_center_1 is None:
					dbgprint(dataloader, LogLevel.WARNING, f'Error getting random tree mask 1 - {ent["image_fn"]}')
				if tree_center_2 is None:
					dbgprint(dataloader, LogLevel.WARNING, f'Error getting random tree mask 2 - {ent["image_fn"]}')
		
				#tree_mask_1 = draw_points_on_image(cv2.cvtColor(tree_mask_1, cv2.COLOR_GRAY2RGB), [list(tree_center_1)])
				#tree_mask_2 = draw_points_on_image(cv2.cvtColor(tree_mask_2, cv2.COLOR_GRAY2RGB), [list(tree_center_2)])
				img = cv2.resize(img, (small_mask.shape[1], small_mask.shape[0]))
				#img = draw_points_on_image(img, [list(tree_center_1), list(tree_center_2)], color=(0, 0, 255), radius=5)
				img = draw_points_on_image(img, [list(reversed(tree_center_1)), list(reversed(tree_center_2))], color=(0, 0, 255), radius=5)
				# Display the extracted tree mask
				cv2.imshow("image", img)
				cv2.imshow("segmentation", seg_mask * 255)
				cv2.imshow("instance", small_mask)
				cv2.imshow("get_random_tree_mask()", tree_mask_1 * 255)
				#cv2.imshow("extract_random_tree_2()" , tree_mask_2 * 255)
				cv2.waitKey(0)
				cv2.destroyAllWindows()




		#classes, freqs	= get_unique_classes  (imask, is_grayscale_img(imask))
		#dbgprint(dataloader, LogLevel.TRACE, f'Classes: {classes} - {freqs}')
		#rgb_mask	= replace_class_colors(imask, classes, freqs=freqs)

		#cv2.imshow("Image", img)
		if self.debug_instance_segmentation_masks:
			outfn = Path('/tmp/spread-out-tmp') / str(Path(ent["image_fn"]).name[:-4]+'instance.jpg')
			dbgprint(dataloader, LogLevel.INFO, f'Writing modified instance segmentation mask: {outfn}')
			cv2.imwrite(outfn, imask)
		#cv2.waitKey()


		#num_samples = 30
		#num_samples = 1

		if self.debug_instance_segmentation_masks:
			### TODO: to debug using always the same image
			imask	= cv2.imread('/tmp/ramdrive/spread-mini/downtown-west/instance_segmentation/Tree10_1721274064.png', cv2.IMREAD_UNCHANGED)


		if False:			# keep every binary mask (including bg colors). Outcome: doesn't work (with crossentropy loss at least)
			n_x = int(torch.sqrt(torch.tensor(self.num_samples * self.width  / self.height)))
			n_y = int(torch.sqrt(torch.tensor(self.num_samples * self.height / self.width)))
			#input_points	= extract_points_outside_region(imask, num_samples, bg_color=[255, 255, 255])
			input_points	= create_asymmetric_point_grid(small_mask.shape[1], small_mask.shape[0], n_x, n_y)
			if isinstance(input_points, np.ndarray) or isinstance(input_points, torch.Tensor):
				dbgprint(dataloader, LogLevel.TRACE, f"Input points shape: {input_points.shape}")
			if isinstance(input_points, list):
				dbgprint(dataloader, LogLevel.TRACE, f"Input points len  : {len(input_points)}")
			dbgprint(dataloader, LogLevel.TRACE, f"Input points: {input_points}")
	
			if self.debug_input_points:
				smask_p	= draw_points_on_image(small_mask, input_points)
				#cv2.imwrite(Path('/tmp/spread-out-tmp') / str(Path(ent["image"]).name[:-4]+'points.jpg'), imask)
				outfn	= str(Path(f'/tmp/small-mask-with-points-{datetime.now():%Y-%m-%d-%H-%M-%S}')) + str(Path(ent["image_fn"]).name[:-4]+'-points.png')
				dbgprint(dataloader, LogLevel.WARNING, f'Writing mask with points: {outfn}')
				cv2.imwrite(outfn, smask_p)
		elif False:
			#inds = np.unique(small_mask)[1:]
			reshaped = small_mask.reshape(-1, 3)
			colors = np.unique(reshaped, axis=0)
			dbgprint(dataloader, LogLevel.TRACE, f'Unique colors: {colors}')
			if len(colors) > 0:
				color_idx = np.random.choice(range(len(colors)))
				color = colors[color_idx]
				dbgprint(dataloader, LogLevel.TRACE, f'Random color idx: {color_idx} - color: {color}')
				mask = (small_mask == color).astype(np.uint8)		# only our selected objects gets binarized in the mask
				dbgprint(dataloader, LogLevel.TRACE, f'Color palette: {self.color_palette_as_list}')
				if list(color) in self.color_palette_as_list:
					# In color_palette there are also walls, roads and similar stuff, I'm appalled by this dataset...
					# really can't find a way to say if it's a tree or not just looking at the color, this dataset is pure hell
					dbgprint(dataloader, LogLevel.TRACE, f'Random color idx: {color_idx} - color: {list(color)} - is in color palette')
					if mask.sum() <= 15000:				# trees should be "small"
						mask[mask==1] = 3			# if it's a tree, it gets +1
					else:
						mask[mask==1] = 2			# larger than 15k px == wall, road, sky, etc. (480x270 images)
				else:
					mask[mask==1] = 1				# this is "true" backgrund (e.g. sky)
					dbgprint(dataloader, LogLevel.TRACE, f'Random color idx: {color_idx} - color: {list(color)} - not in color palette')
				coords = np.argwhere(mask > 0)
				yx = coords[np.random.randint(len(coords))]
				point = [[yx[1], yx[0]]]
			else:
				dbgprint(dataloader, LogLevel.ERROR, f'Only background in small_mask: {small_mask}')
				return None, None, None

			if self.debug_input_points:
				smask_p	= draw_points_on_image(small_mask, point)
				outfn	= str(Path(f'/tmp/small-mask-all-instances-with-points-{datetime.now():%Y-%m-%d-%H-%M-%S}')) + str(Path(ent["image_fn"]).name[:-4]+'-points.png')
				dbgprint(dataloader, LogLevel.WARNING, f'Writing colored mask with 1 point: {outfn} - point: {point}')
				cv2.imwrite(outfn, smask_p[..., ::-1])

				smask_p	= draw_points_on_image(mask, point)
				outfn	= str(Path(f'/tmp/small-mask-only-one-instance-with-points-{datetime.now():%Y-%m-%d-%H-%M-%S}')) + str(Path(ent["image_fn"]).name[:-4]+'-points.png')
				dbgprint(dataloader, LogLevel.WARNING, f'Writing binary mask with 1 point: {outfn} - point: {point} - color: {color}')
				cv2.imwrite(outfn, smask_p[..., ::-1])
			dbgprint(dataloader, LogLevel.TRACE, f'{type(img)} {type(mask)} {type(point)}')
			dbgprint(dataloader, LogLevel.TRACE, f"Input points len: {len(point)}")
			dbgprint(dataloader, LogLevel.TRACE, f"Input points: {point}")

			# we tried to do all the instances at once... we failed. Let's try with the original (dumb) method
			return img, imask, point, mask, color_ids
		else:
			return img, imask, tree_centers, tree_mask, color_ids
			#return img, imask, [tree_center], tree_mask, color_ids


		#input_points	= np.ravel(input_points)

		#dbgprint(dataloader, LogLevel.INFO, f"Image shape	  : {Img.shape}")
		#dbgprint(dataloader, LogLevel.INFO, f"Mask  shape	  : {ann_map.shape}")
		#dbgprint(dataloader, LogLevel.INFO, f"RGB mask shape   : {rgb_mask.shape}")


		'''
		# Process the annotation map
		mat_map = ann_map[:, :, 0]
		ves_map = ann_map[:, :, 2]
		mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)

		inds = np.unique(mat_map)[1:]
		if len(inds) > 0:
			ind = np.random.choice(inds)
			mask = (mat_map == ind).astype(np.uint8)
			coords = np.argwhere(mask > 0)
			yx = coords[np.random.randint(len(coords))]
			point = [[yx[1], yx[0]]]
		else:
			mask = np.zeros_like(mat_map, dtype=np.uint8)
			point = [[0, 0]]  # Provide a default point

		return Img, mask, point
		'''

		#return img, ann_map, input_points
		return img, imask, input_points, small_mask, color_ids


