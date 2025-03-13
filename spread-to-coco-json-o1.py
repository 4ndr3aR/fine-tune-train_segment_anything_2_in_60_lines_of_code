#!/usr/bin/env python3

'''
import random

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from instance_seg_loss import read_color_palette, read_colorid_file

from utils import draw_points_on_image
'''

import os
import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as mask_utils

from dbgprint import dbgprint
from dbgprint import *

from datasets.spread import SpreadDataset, get_all_trees, get_bbox, show_instances

# Suppose we have this function from your code that returns: 
# [(tree_mask, center_point, bbox, color, nonzero, tree_trunk, largest_blob), ...]

def print_stats(stats_by_subcat, ratio_threshold):
	# Finally, print the statistics
	print("\n===== Dataset Statistics =====")
	for split_name in ["train", "val", "test"]:
		if split_name not in stats_by_subcat:
			continue
		print(f"\n--- {split_name.upper()} split ---")
		for subcat, st in stats_by_subcat[split_name].items():
			n_files		= st["n_files"]
			n_bboxes	= st["n_bboxes"]
			avg_bbox_area	= (st["sum_bbox_area"] / n_bboxes) if n_bboxes > 0 else 0.0
			avg_mask_px	= (st["sum_mask_px"] / n_bboxes) if n_bboxes > 0 else 0.0
			avg_largest_blob= (st["sum_largest_blob"] / n_bboxes) if n_bboxes > 0 else 0.0
			ratio_exceed	= st["count_ratio_exceed"]
			avg_density	= (st["sum_density"] / n_bboxes) if n_bboxes > 0 else 0.0
			avg_tree_trunk	= (st["sum_tree_trunk"] / n_bboxes) if n_bboxes > 0 else 0.0
			# If we wanted average W x H specifically
			#   we can store the average of W and average of H, or just do sqrt of avg_bbox_area, etc.
			#   But let's do average W*H as is, which is avg_bbox_area above.

			print(
				f"Subcategory: {subcat:20s} | "
				f"Files: {n_files:3d}, "
				f"BBoxes: {n_bboxes:5d}, "
				f"Avg BBoxArea (WxH): {avg_bbox_area:8.2f}, "
				f"Avg Mask px: {avg_mask_px:8.2f}, "
				f"Avg LargestBlob: {avg_largest_blob:8.2f}, "
				f"Count Ratio>({ratio_threshold}): {ratio_exceed:5d}, "
				f"Avg Density: {avg_density:6.3f}, "
				f"Avg TreeTrunk px: {avg_tree_trunk:8.2f}"
			)



def write_coco_annotations(
	dataset_splits,
	output_dir,
	ratio_threshold=2.0,   # for counting the number of bboxes with width/height > ratio_threshold
	px_threshold=50,
	px_threshold_perc=0.01,
	debug_instance_segmentation_masks=True,
):
	"""
	Writes COCO-style JSON files (train.json, val.json, test.json) for a dataset
	given a dictionary of splits (each being a list of dicts with keys image_fn, segmentation_fn, instance_fn).
	
	Also prints statistics per subcategory:
	  - number of files processed
	  - number of bboxes (mask instances) written
	  - average size of the bbox (width x height)
	  - average size of the mask in px
	  - average size of the largest blob
	  - number of bboxes with width/height > ratio_threshold
	  - average density (mask px / (width x height))
	  - average size of the tree trunk
	"""

	# Prepare output directory
	os.makedirs(output_dir, exist_ok=True)

	# COCO structure template
	# We'll do one structure for each split (train, val, test)
	template_coco = {
		"info": {
			"description": "Trees dataset in COCO format",
			"url": "",
			"version": "1.0",
			"year": 2023,
			"contributor": "",
			"date_created": ""
		},
		"licenses": [],
		"images": [],
		"annotations": [],
		"categories": [
			{
				"id": 1,
				"name": "tree",
				"supercategory": "nature"
			}
		]
	}

	# We’ll track separate image/annotation IDs for each split
	# (Alternatively, you could track them globally if you prefer.)
	splits_coco = {
		"train": {
			"images": [],
			"annotations": [],
			"categories": template_coco["categories"],
			"info": template_coco["info"],
			"licenses": template_coco["licenses"],
		},
		"val": {
			"images": [],
			"annotations": [],
			"categories": template_coco["categories"],
			"info": template_coco["info"],
			"licenses": template_coco["licenses"],
		},
		"test": {
			"images": [],
			"annotations": [],
			"categories": template_coco["categories"],
			"info": template_coco["info"],
			"licenses": template_coco["licenses"],
		}
	}

	# For ID tracking (simple counters)
	image_id_counters = {"train": 1, "val": 1, "test": 1}
	annotation_id_counters = {"train": 1, "val": 1, "test": 1}

	# We also want to accumulate stats by subcategory
	# For each subcategory, we'll keep a dict to store aggregates.
	# Key = subcat name, value = dictionary of stats
	stats_by_subcat = {
		"train": {},
		"val": {},
		"test": {}
	}

	def derive_subcategory_from_path(img_path: str) -> str:
		"""
		Parse the path or name to figure out which subcategory 
		it belongs to. This is a placeholder; adjust to your actual naming.
		"""
		# Example: if path is ".../birch-forest/filename.jpg"
		# subcategory might be the parent folder's name.
		# Or you can parse any other convention you have.

		p = Path(img_path)
		# Possibly the subcategory is the parent folder
		subcat = p.parent.name  
		return subcat

	# Helper function to initialize subcat stats if not present
	def init_subcat_stats(stats_dict, subcat):
		if subcat not in stats_dict:
			stats_dict[subcat] = {
				"n_files": 0,
				"n_bboxes": 0,
				"sum_bbox_area": 0.0,
				"sum_mask_px": 0.0,
				"sum_largest_blob": 0.0,
				"count_ratio_exceed": 0,
				"sum_density": 0.0,
				"sum_tree_trunk": 0.0,
				# We also keep track of total width and height separately 
				# to compute average WxH if needed
				"sum_w": 0.0,
				"sum_h": 0.0
			}

	# Iterate over each split
	for split_name in ["train", "val", "test"]:
		entries = dataset_splits.get(split_name, [])
		# If there are no entries for this split, just skip
		if len(entries) == 0:
			continue

		for entry_idx, entry in enumerate(entries):
			img_fn  = entry["image_fn"]
			seg_fn  = entry["segmentation_fn"]
			iseg_fn = entry["instance_fn"]

			# Read your image to get height/width for COCO 'images' entry
			# (This is important for the COCO format)
			# If your images are large, you might want to do something more memory-efficient, 
			# but here's the straightforward approach:
			img = cv2.imread(img_fn)
			if img is None:
				print(f"Warning: could not read {img_fn}. Skipping.")
				continue

			height, width = img.shape[:2]
			
			# Create an 'images' record
			this_image_id = image_id_counters[split_name]
			splits_coco[split_name]["images"].append({
				"file_name": str(img_fn),
				"height": height,
				"width": width,
				"id": this_image_id
			})
			image_id_counters[split_name] += 1

			# Determine subcategory for stats
			subcat = derive_subcategory_from_path(img_fn)
			init_subcat_stats(stats_by_subcat[split_name], subcat)
			stats_by_subcat[split_name][subcat]["n_files"] += 1

			# Read segmentation, instance if needed for get_all_trees
			seg_mask  = cv2.imread(seg_fn , cv2.IMREAD_GRAYSCALE)
			iseg_mask = cv2.imread(iseg_fn, cv2.IMREAD_UNCHANGED)  # possibly 3-channel or single channel, adapt

			# we want segmentation masks to be 480x270 to draw coords on trunks and then pick the color from instance segmentation masks
			seg_mask  = cv2.resize(seg_mask,   (iseg_mask.shape[1], iseg_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

			dbgprint(dataloader, LogLevel.TRACE, f'{seg_mask.shape = } - {iseg_mask.shape = } - {seg_mask.dtype = } - {iseg_mask.dtype = }')

			if seg_mask is None or iseg_mask is None:
				print(f"Warning: could not read seg or instance mask for {img_fn}. Skipping.")
				continue

			# If your instance mask is color-coded, you can keep it or transform to labels, etc.
			# We'll treat iseg_mask as "small_mask"
			#small_mask = (iseg_mask > 0).astype(np.uint8)
			small_mask = iseg_mask

			# Call get_all_trees(...) to get your trees
			all_the_trees = get_all_trees(
				seg_mask, 
				small_mask, 
				px_threshold=px_threshold, 
				px_threshold_perc=px_threshold_perc
			)

			if debug_instance_segmentation_masks:
				show_instances(img_fn, img, small_mask, seg_mask, all_the_trees)


			# For each tree in this image, create a COCO 'annotation' record
			for (tree_mask, center_point, bbox, color, nonzero, tree_trunk, largest_blob) in all_the_trees:
				# bbox is (x, y, w, h)
				x, y, w, h = bbox
				# The mask is binary (0/1 or 0/255).  We convert to RLE.
				# We'll assume it's a 2D mask with the same shape as seg_mask.
				# Make sure it's Fortran-contiguous for pycocotools
				# If tree_mask is boolean array of shape [H, W], do:
				encoded_mask = mask_utils.encode(
					np.asfortranarray(tree_mask.astype(np.uint8))
				)
				# Convert byte-string to ascii so it can be serialized in json
				encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")

				# Build the annotation
				this_annotation_id = annotation_id_counters[split_name]
				annotation_id_counters[split_name] += 1

				# COCO typically uses [x, y, width, height] for bounding box
				# area can be len(mask.nonzero())
				area = float(np.sum(tree_mask))  # number of pixels in mask

				annotation_record = {
					"id": this_annotation_id,
					"image_id": this_image_id,
					"category_id": 1,  # "tree"
					"segmentation": encoded_mask,
					"bbox": [int(x), int(y), int(w), int(h)],
					"area": area,
					"iscrowd": 0
				}

				# Add to the corresponding split COCO list
				splits_coco[split_name]["annotations"].append(annotation_record)

				# Update stats
				stats_by_subcat[split_name][subcat]["n_bboxes"] += 1
				# sum_bbox_area is storing the sum of (w*h)
				stats_by_subcat[split_name][subcat]["sum_bbox_area"] += (w * h)
				# sum_mask_px is storing the total of mask's pixel counts
				stats_by_subcat[split_name][subcat]["sum_mask_px"] += area
				# sum_largest_blob is the total area of largest_blob
				lb_area = float(np.sum(largest_blob)) if largest_blob is not None else 0.0
				stats_by_subcat[split_name][subcat]["sum_largest_blob"] += lb_area
				# sum of tree trunk area
				tk_area = float(np.sum(tree_trunk)) if tree_trunk is not None else 0.0
				stats_by_subcat[split_name][subcat]["sum_tree_trunk"] += tk_area

				# Count ratio
				if h > 0 and (w / h) > ratio_threshold:
					stats_by_subcat[split_name][subcat]["count_ratio_exceed"] += 1

				# sum_density = sum of (area / (w*h))
				if w > 0 and h > 0:
					stats_by_subcat[split_name][subcat]["sum_density"] += area / (w * h)

				# For average bounding box dimension if you want separate W and H
				stats_by_subcat[split_name][subcat]["sum_w"] += w
				stats_by_subcat[split_name][subcat]["sum_h"] += h

			# Print the stats
			#print_stats(stats_by_subcat, ratio_threshold)
			if entry_idx % 1000 == 0:
				print(f'.', end="", flush=True)

	print(f'Done!\n', flush=True)

	# Now write out each split's result
	for split_name in ["train", "val", "test"]:
		if len(splits_coco[split_name]["images"]) == 0:
			continue
		out_coco = {
			"info": splits_coco[split_name]["info"],
			"licenses": splits_coco[split_name]["licenses"],
			"categories": splits_coco[split_name]["categories"],
			"images": splits_coco[split_name]["images"],
			"annotations": splits_coco[split_name]["annotations"]
		}
		json_filename = os.path.join(output_dir, f"{split_name}.json")
		with open(json_filename, "w") as f:
			json.dump(out_coco, f, indent=2)
		print(f"Wrote {json_filename}")

	print_stats(stats_by_subcat, ratio_threshold)


# -----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
	# Suppose you have a dictionary with your data splits already established
	# Each entry has "image_fn", "segmentation_fn", "instance_fn"
	# The ratio of train / val / test is presumably 70/20/10,
	# but we assume you have already done that somewhere else.

	dbgprint(main, LogLevel.INFO, "Loading Spread dataset...")
	data_dir = Path("/mnt/raid1/dataset/spread/spread")
	dataset  = SpreadDataset(data_dir, split="all", preload=False)
	train, valid, test = dataset.train_data, dataset.val_data, dataset.test_data
	dbgprint(main, LogLevel.INFO, f"Loaded {len(train)}, {len(valid)}, {len(test)} images. Total: {len(train) +  len(valid) + len(test)} images.")

	'''
	your_dataset_splits = {
		"train": [
			{
				"image_fn": "data/birch-forest/img1.jpg",
				"segmentation_fn": "data/birch-forest/seg1.png",
				"instance_fn": "data/birch-forest/inst1.png"
			},
			{
				"image_fn": "data/broadleaf-forest/img2.jpg",
				"segmentation_fn": "data/broadleaf-forest/seg2.png",
				"instance_fn": "data/broadleaf-forest/inst2.png"
			},
			# ...
		],
		"val": [
			# ...
		],
		"test": [
			# ...
		]
	}
	'''

	# Call the function
	write_coco_annotations(
		dataset_splits={'train': train, 'val': valid, 'test': test},
		output_dir="./coco_output",
		ratio_threshold=2.0,
		px_threshold=50,
		px_threshold_perc=0.01
	)











