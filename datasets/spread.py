import os

import cv2

import random

import numpy as np

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


def get_all_trees(seg_mask, iseg_mask, px_threshold=-1, px_threshold_perc=-1):
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

	for color in unique_colors:
		# 3. Create binary mask for each color (tree instance)
		tree_mask = np.all(iseg_mask == color, axis=-1).astype(np.uint8)
		
		# 4. Calculate bounding box using OpenCV
		x, y, w, h = cv2.boundingRect(tree_mask)

		bbox = (x, y, x + w, y + h)

		# 5. Calculate tree center (centroid) from the mask
		coords = np.column_stack(np.where(tree_mask == 1))
		if len(coords) > 0:
			center = coords.mean(axis=0).astype(int)
			center_point = (center[0], center[1])  # (row, col)
		else:
			center_point = None
			dbgprint(dataloader, LogLevel.WARNING, f'get_all_trees() - No coordinates found for tree with color: {color}')
			continue

		if seg_mask[center_point[0], center_point[1]] == 0:
			dbgprint(dataloader, LogLevel.WARNING,   f'get_all_trees() - center_point not in trunk_coords: {center_point} - {trunk_coords}')
			continue

		nonzero = np.count_nonzero(tree_mask)

		# 6. Store result
		if nonzero > px_threshold and nonzero > px_threshold_perc * iseg_mask.shape[0] * iseg_mask.shape[1] / 100.0:
			results.append((tree_mask, center_point, bbox, color, nonzero))
		else:
			dbgprint(dataloader, LogLevel.WARNING,   f'get_all_trees() - nonzero = {nonzero} - px_threshold = {px_threshold} - px_threshold_perc = {px_threshold_perc * iseg_mask.shape[0] * iseg_mask.shape[1] / 100.0}')

	return results





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
		self.debug_instance_segmentation_masks	= False
		self.debug_semantic_segmentation_masks	= False
		self.debug_input_points			= False

		if SpreadDataset._data is None:
			SpreadDataset._data = self._load_data(data_dir)

		self.data				= self._split_data(SpreadDataset._data, split, train_ratio, val_ratio, test_ratio)
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
						"image"		 : cv2.imread(im_fn)[..., ::-1] if self.preload else None,	# RGB instead of BGRA plz
						"instance"	 : cv2.imread(imask_fn, cv2.IMREAD_UNCHANGED)[..., ::-1] if self.preload else None, # RGB plz
						"segmentation"	 : cv2.imread(smask_fn, cv2.IMREAD_UNCHANGED) if self.preload else None,            # Grayscale
						"colorids"	 : read_colorid_file(colorid_fn),
					})
		print(' done!', flush=True)

		#dbgprint(dataloader, LogLevel.INFO, f"Total number of entries: {len(self.data)}")
		dbgprint(dataloader, LogLevel.INFO, f"\nLoaded {len(data)} images")

		return data

	def _split_data(self, data, split, train_ratio, val_ratio, test_ratio):
		# Create splits
		train_data, test_data = train_test_split(data,  test_size=test_ratio, random_state=42)
		train_data, val_data  = train_test_split(train_data, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

		dbgprint(dataloader, LogLevel.TRACE, f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")			# we already print this later
		
		if split == "train":
			return train_data
		elif split == "val":
			return val_data
		elif split == "test":
			return test_data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		ent	= self.data[idx]
		dbgprint(dataloader, LogLevel.TRACE, f'Reading images: {ent["image"]} - {ent["instance"]} - {ent["segmentation"]}')
		#img	= cv2.imread(ent["image"])[..., ::-1]					# Convert BGR to RGB
		img = None
		if ent["image"] is not None:
			img = ent["image"]
		else:
			dbgprint(dataloader, LogLevel.TRACE, f'Reading image: {ent["image_fn"]}')
			img = cv2.imread(ent["image_fn"])[..., ::-1]
		dbgprint(dataloader, LogLevel.TRACE, f"------------------- Image shape: {img.shape}")
		if img is None:
			dbgprint(dataloader, LogLevel.ERROR, f'Error reading image: {ent["image"]}')
		#ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_UNCHANGED)			# Read as is
		#imask = cv2.imread(ent["instance"], cv2.IMREAD_GRAYSCALE)			# Read grayscale
		#imask	= cv2.imread(ent["instance"],		cv2.IMREAD_UNCHANGED)		# Read as is
		small_mask = ent["instance"] if ent["instance"] is not None else cv2.imread(ent["instance_fn"], cv2.IMREAD_UNCHANGED)[..., ::-1]
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
		#smask   = cv2.resize(smask,      (self.width, self.height), interpolation=cv2.INTER_NEAREST)

		# we want segmentation masks to be 480x270 to draw coords on trunks and then pick the color from small_masks (instance segmentation masks)
		seg_mask = cv2.resize(seg_mask,   (small_mask.shape[1], small_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
		#cv2.imshow("Segmentation resized", seg_mask * 255)



		all_the_trees = get_all_trees(seg_mask, small_mask)				# a list of (tree_mask, center_point, bbox)
		for idx, (tree_mask, center_point, bbox, color, nonzero) in enumerate(all_the_trees):	# where bbox = (x, y, w, h)
			x, y, w, h = bbox
			dbgprint(dataloader, LogLevel.WARNING, f'get_all_trees[{idx}] - tree mask shape: {tree_mask.shape} - center point: {center_point} - bbox: {bbox} - color: {color} - nonzero: {nonzero}')
			img2 = cv2.resize(img, (small_mask.shape[1], small_mask.shape[0]))
			img2 = draw_points_on_image(img2, [list(reversed(center_point))], color=(0, 0, 255), radius=5)
			img2 = cv2.rectangle(img2, (x, y), (w, h), color=(255, 0, 0), thickness=2)
			print(f'tree_mask.shape: {tree_mask.shape}')
			colored_tree_mask = cv2.bitwise_and(small_mask, small_mask, mask=tree_mask)
			#colored_tree_mask = small_mask[tree_mask]
			#colored_tree_mask = small_mask[tree_mask == 1]
			#colored_tree_mask = colored_tree_mask.reshape((small_mask.shape[0], small_mask.shape[1], 3))
			print(f'colored_tree_mask.shape: {colored_tree_mask.shape}')
			# Display the extracted tree mask
			cv2.imshow("image",			img2)
			cv2.imshow("segmentation",		seg_mask * 255)
			cv2.imshow("instance",			small_mask)
			cv2.imshow("colored_tree mask",		colored_tree_mask)
			cv2.imshow("get_all_trees[{idx}]",	tree_mask * 255)
			cv2.waitKey(0)
			cv2.destroyAllWindows()



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
			return img, imask, [tree_center], tree_mask, color_ids


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


