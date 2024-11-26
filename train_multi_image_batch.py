#!/usr/bin/env python3

import os
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(description="Configuration for your script")

	parser.add_argument("--gpu_id",		type=int,	default=0,		help="GPU ID to use (default: 0)")
	parser.add_argument("--num_workers",	type=int,	default=16,		help="Number of worker threads (default: 16)")
	parser.add_argument("--num_epochs",	type=int,	default=1000,		help="Number of training epochs (default: 1000)")
	parser.add_argument("--batch_size",	type=int,	default=63,		help="Batch size for training (default: 63)")
	parser.add_argument("--model_size",	type=str,	default='small',	help="Model size (default: small)", choices=['tiny', 'small', 'base', 'large'])
	parser.add_argument("--dataset_name",	type=str,	default="LabPicsV1",	help="Path to the dataset directory")
	parser.add_argument("--use_wandb",	action="store_true",			help="Enable Weights & Biases logging (default: False)")
	parser.add_argument("--lr",		type=float,	default=1e-5,		help="Learning rate (default: 1e-5)")
	parser.add_argument("--wr",		type=float,	default=4e-5,		help="Weight regularization rate (default: 4e-5)")
	parser.add_argument("--split",		type=str,	default='70/20/10',	help="Train/validation/test split ratio (default: 70/20/10)")
	parser.add_argument("--img_resolution", type=str,	default='960x540',	help="Image resolution in widthxheight format (default: 960x540)")

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

	return args

if __name__ == "__main__":
	args = parse_arguments()		# get the GPU to use, then import torch and everything else...

from pathlib import Path

import numpy as np
import torch

import pdb
import cv2

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dbgprint import dbgprint
from dbgprint import *

def unpack_resolution(resolution_str):
    """Unpacks the resolution string into width and height."""
    try:
        width, height = map(int, resolution_str.split('x'))
    except ValueError:
        raise ValueError("Image resolution should be in 'widthxheight' format (e.g., 960x540)")

    return width, height

def decode_split_ratio(split_str):
	"""Decodes the split ratio string and returns train, validation, and test ratios."""
	ratios = split_str.split('/')
	if len(ratios) != 3:
		raise ValueError("Split ratio should be in 'train/val/test' format (e.g., 70/20/10)")

	try:
		train_ratio = int(ratios[0]) / 100
		val_ratio = int(ratios[1]) / 100
		test_ratio = int(ratios[2]) / 100
	except ValueError:
		raise ValueError("Split ratios must be integers")

	dbgprint(dataloader, LogLevel.TRACE, f'Split ratios: {train_ratio}, {val_ratio}, {test_ratio} = {round(train_ratio + val_ratio + test_ratio, 3)}')
	if round(train_ratio + val_ratio + test_ratio, 3) != 1.0:
		raise ValueError("Split ratios must sum to 100")

	return train_ratio, val_ratio, test_ratio

def init_wandb(use_wandb, args, project_name):
	"""Initializes Weights & Biases if enabled."""
	if use_wandb:
		wandb.init(project=project_name, config=args)	# Replace "your-project-name"

def set_model_paths(model_size):
    """Sets the checkpoint and configuration file paths based on the model size."""
    if model_size == 'tiny':
        sam2_checkpoint		= "sam2_hiera_tiny.pt"		# path to model weight
        model_cfg		= "sam2_hiera_t.yaml"		# model config
    elif model_size == 'small':
        sam2_checkpoint		= "sam2_hiera_small.pt"
        model_cfg		= "sam2_hiera_s.yaml"
    elif model_size == 'base':
        sam2_checkpoint		= "sam2_hiera_base.pt"
        model_cfg		= "sam2_hiera_b.yaml"
    elif model_size == 'large':
        sam2_checkpoint		= "sam2_hiera_large.pt"
        model_cfg		= "sam2_hiera_l.yaml"
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    return sam2_checkpoint, model_cfg



# Dataset class for LabPicsV1 dataset
class LabPicsDataset(Dataset):
    def __init__(self, data_dir, split="Train"):
        self.data_dir = data_dir
        self.split = split
        self.data = []
        for name in os.listdir(os.path.join(data_dir, "Simple", self.split, "Image")):
            self.data.append({
                "image": os.path.join(data_dir, "Simple", self.split, "Image", name),
                "annotation": os.path.join(data_dir, "Simple", self.split, "Instance", name[:-4] + ".png")
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        Img = cv2.imread(ent["image"])[..., ::-1]
        ann_map = cv2.imread(ent["annotation"])

        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

        Img = np.pad(Img, ((0, 1024 - Img.shape[0]), (0, 1024 - Img.shape[1]), (0, 0)), 'constant')
        ann_map = np.pad(ann_map, ((0, 1024 - ann_map.shape[0]), (0, 1024 - ann_map.shape[1]), (0, 0)), 'constant')

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

        dbgprint(dataloader, LogLevel.TRACE, f'{type(Img)} {type(mask)} {type(point)}')
        return Img, mask, point

def collate_fn(data):
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 1. - {type(data)    = } {len(data)    = }')
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 2. - {type(data[0]) = } {len(data[0]) = }')
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 3. - {type(data[1]) = } {len(data[1]) = }')
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 4. - {type(data[2]) = } {len(data[2]) = }')
    imgs, masks, points = zip(*data)
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 5. - {type(imgs)    = } {len(imgs)    = }')
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 6. - {type(masks)   = } {len(masks)   = }')
    dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 7. - {type(points)  = } {len(points)  = }')
    return list(imgs), list(masks), list(points)

class SpreadDataset(Dataset):
    def __init__(self, data_dir, split="train", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_dir = data_dir
        self.split = split
        
        # Collect all data entries
        self.data = []
        for class_dir in os.listdir(data_dir):
            dbgprint(dataloader, LogLevel.TRACE, f'Reading directory: {class_dir}')
            if not os.path.isdir(os.path.join(data_dir, class_dir)):
                continue
            rgb_dir = os.path.join(data_dir, class_dir, "rgb")
            dbgprint(dataloader, LogLevel.TRACE, f'RGB directory: {rgb_dir}')
            if not os.path.isdir(rgb_dir) or not os.path.exists(rgb_dir):
                continue
            instance_dir = os.path.join(data_dir, class_dir, "instance_segmentation")
            dbgprint(dataloader, LogLevel.TRACE, f'Instance segmentation directory: {instance_dir}')
            if not os.path.isdir(instance_dir) or not os.path.exists(instance_dir):
                continue
            for name in os.listdir(rgb_dir):
                if name.endswith(".png"):
                    self.data.append({
                        "image": os.path.join(rgb_dir, name),
                        "annotation": os.path.join(instance_dir, name)
                    })

        dbgprint(dataloader, LogLevel.INFO, f"Total number of entries: {len(self.data)}")

        # Create splits
        train_data, test_data = train_test_split(self.data,  test_size=test_ratio, random_state=42)
        train_data, val_data  = train_test_split(train_data, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)
        
        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        dbgprint(dataloader, LogLevel.TRACE, f'Reading images: {ent["image"]} - {ent["annotation"]}')
        Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
        ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_UNCHANGED)  # Read as is

        # Resize images and masks to the same resolution
        Img = cv2.resize(Img, (960, 540))
        ann_map = cv2.resize(ann_map, (960, 540), interpolation=cv2.INTER_NEAREST)

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
GPU_ID = 0
NUM_WORKERS = 16					# Set number of threads globally
NUM_EPOCHS  = 1000
batch_size = 63
model_size = 'small'					# 'tiny' | 'small' | 'base' | 'large' - also write the function to set sam2_checkpoint/model_cfg variables to sam2_hiera_small.pt/sam2_hiera_s.yaml accordingly
dataset_dir = "/mnt/raid1/dataset/LabPicsV1"		# This should be a pathlib Path
use_wandb = False					# Also write the function to init wandb
LR = 1e-5
WR = 4e-5
SPLIT = '70/20/10'					# Also write the function to decode this and return train_ratio, val_ratio, test_ratio
IMAGE_RESOLUTION = '960x540'				# Also write the function to unpack this into w, h tuple (e.g. (960, 540))
'''

def reset_breakpoints(disabled=[]):
    global _breakpoints
    _breakpoints = dict((x, False) for x in disabled)

def set_breakpoint(tag, condition=True):
	# Use with:

	# set_breakpoint('mycode0')
	# set_breakpoint('mycode1', x == 4)

	if tag not in _breakpoints:
		_breakpoints[tag] = True
		if condition:
			pdb.set_trace()
	else:
		if _breakpoints[tag] and condition:
			pdb.set_trace()







def sam2_predict(predictor, image, mask, input_point, input_label, box=None, mask_logits=None, normalize_coords=True):
	dbgprint(predict, LogLevel.TRACE, f'1. - {type(image)    = } - {type(mask)     = } - {type(input_point) = }')
	dbgprint(predict, LogLevel.TRACE, f'2. - {type(image)    = } - {len(image)     = }')
	dbgprint(predict, LogLevel.TRACE, f'3. - {type(image[0]) = } - {image[0].shape = } - {image[0].dtype = }')
	dbgprint(predict, LogLevel.TRACE, f'4. - {type(mask[0])  = } - {mask[0].shape  = } - {mask[0].dtype = }')

	predictor.set_image_batch(image)				# apply SAM image encoder to the image

	mask_input, unnorm_coords, labels, unnorm_box	= predictor._prep_prompts(input_point, input_label, box=box, mask_logits=mask_logits, normalize_coords=normalize_coords)
	sparse_embeddings, dense_embeddings		= predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None if box is None else unnorm_box, masks=None if mask_logits is None else mask_input)

	# mask decoder

	high_res_features				= [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
	low_res_masks, prd_scores, _, _			= predictor.model.sam_mask_decoder(
								image_embeddings=predictor._features["image_embed"],
								image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
								sparse_prompt_embeddings=sparse_embeddings,
								dense_prompt_embeddings=dense_embeddings,
								multimask_output=True,
								repeat_image=False,
								high_res_features=high_res_features,)

	dbgprint(predict, LogLevel.TRACE, f'5. - {low_res_masks.shape = } - {len(predictor._orig_hw) = }')
	# Upscale the masks to the original image resolution
	prd_masks					= predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

	return prd_masks, prd_scores

def calc_loss_and_metrics(mask, prd_masks, prd_scores, score_loss_weight=0.05):
	# Segmentaion Loss caclulation

	gt_mask  = torch.tensor(np.array(mask).astype(np.float32)).cuda()
	prd_mask = torch.sigmoid(prd_masks[:, 0])						# Turn logit map to probability map
	seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

	# Score loss calculation (intersection over union) IOU

	inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
	iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
	score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
	loss = seg_loss + score_loss*score_loss_weight						# mix losses

	return loss, seg_loss, score_loss, iou


# Validation function
def validate(predictor, val_loader):
	total_loss = 0
	total_iou = 0
	total_seg_loss = 0
	total_score_loss = 0
	count = 0
	predictor.model.eval()
	with torch.no_grad():
		for image, mask, input_point in val_loader:

			input_point = torch.tensor(np.array(input_point)).cuda().float()
			input_label = torch.ones(input_point.shape[0], 1).cuda().float()		# create labels

			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'1. - {type(image)    = } - {type(mask)     = } - {type(input_point) = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'2. - {len(image)     = } - {len(mask)      = } - {len(input_point) = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'3. - {type(image[0]) = } - {image[0].shape = } - {image[0].dtype = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'4. - {type(mask[0])  = } - {mask[0].shape  = } - {mask[0].dtype = }')

			prd_masks, prd_scores		= sam2_predict(predictor, image, mask, input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
			loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(mask, prd_masks, prd_scores, score_loss_weight=0.05)

			if count == 0 and False:
				#wandb_images = wandb.Image(image, caption="Validation 1st batch")
				#wandb.log({"examples": images})
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'5. - {type(image) = } - {type(mask) = } - {type(prd_masks) = } - {type(prd_scores) = }')
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'6. - {len(image)  = } - {len(mask)  = } - {prd_masks.shape = } - {prd_scores.shape = }')

				table = wandb.Table(columns=["Image", "Mask"])

				class_labels = {1: "class1", 2: "class2"}
				for img, msk, prd_mask, prd_score in zip(images, mask, prd_masks, prd_scores):
					gt_msk   = torch.tensor(np.array(msk).astype(np.float32)).cuda()
					prd_msk  = torch.sigmoid(prd_mask[:, 0])
					dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'7. - {gt_msk.shape = } - {prd_mask.shape = } - {prd_score.shape = }')
					dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'8. - {gt_msk = }')
					dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'9. - {prd_msk = }')
					dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'0. - {prd_score = }')

					mask_img = wandb.Image(img, masks={
										"prediction":	{"mask_data": prd_msk, "class_labels": class_labels},
										"ground_truth":	{"mask_data": gt_msk,  "class_labels": class_labels}
									})
					table.add_data(img, prd_score[:, 0])
				wandb.log({"Table": table})



			total_loss += loss.item()
			total_iou += iou.mean().item()
			total_score_loss += score_loss.item()
			total_seg_loss += seg_loss.item()
			count += 1

			avg_val_loss = total_loss / count
			avg_val_iou = total_iou / count
			avg_val_score_loss = total_score_loss / count
			avg_val_seg_loss = total_seg_loss / count

			dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'Batch {count}, Validation loss: {avg_val_loss:.4f}, Val IOU: {avg_val_iou:.4f}, Val score loss: {avg_val_score_loss:.4f}, Val seg loss: {avg_val_seg_loss:.4f}')

	return avg_val_loss, avg_val_iou, avg_val_score_loss, avg_val_seg_loss, count





if __name__ == "__main__":
	args		= parse_arguments()

	# Example usage of the functions
	gpu_id			= args.gpu_id
	num_workers		= args.num_workers
	num_epochs		= args.num_epochs
	batch_size		= args.batch_size
	model_size		= args.model_size
	dataset_name		= args.dataset_name
	use_wandb		= args.use_wandb
	lr			= args.lr
	wr			= args.wr
	split			= args.split
	image_resolution	= args.img_resolution
	_breakpoints		= {}

	if use_wandb:
		import wandb

	sam2_checkpoint, model_cfg		= set_model_paths(model_size)
	train_ratio, val_ratio, test_ratio	= decode_split_ratio(split)
	width, height				= unpack_resolution(image_resolution)

	init_wandb(use_wandb, args, project_name=f"SAM2-{model_size}-{dataset_name}-bs-{batch_size}-lr-{lr}-wr-{wr}-imgsz-{width}x{height}")
	torch.set_num_threads(num_workers)

	dbgprint(main, LogLevel.INFO, f"Using the following configuration:")
	dbgprint(main, LogLevel.INFO, f"==================================")
	dbgprint(main, LogLevel.INFO, f"Using GPU ID		: {gpu_id}")
	dbgprint(main, LogLevel.INFO, f"Number of Workers	: {num_workers}")
	dbgprint(main, LogLevel.INFO, f"Number of Epochs		: {num_epochs}")
	dbgprint(main, LogLevel.INFO, f"Batch Size		: {batch_size}")
	dbgprint(main, LogLevel.INFO, f"Model Size		: {model_size}")
	dbgprint(main, LogLevel.INFO, f"Dataset			: {dataset_name}")
	dbgprint(main, LogLevel.INFO, f"Use Wandb		: {use_wandb}")
	dbgprint(main, LogLevel.INFO, f"Learning Rate		: {lr}")
	dbgprint(main, LogLevel.INFO, f"Weight Regularization	: {wr}")
	dbgprint(main, LogLevel.INFO, f"Train/Val/Test Ratio	: {train_ratio}/{val_ratio}/{test_ratio}")
	dbgprint(main, LogLevel.INFO, f"Image Resolution		: {width}x{height}")
	dbgprint(main, LogLevel.INFO, f"SAM2 Checkpoint		: {sam2_checkpoint}")
	dbgprint(main, LogLevel.INFO, f"Model Config		: {model_cfg}")
	dbgprint(main, LogLevel.INFO, f"==================================")

	train_loader	= None
	val_loader	= None
	test_loader	= None

	# Data Loaders
	if "labpic" in dataset_name.lower():
		dbgprint(main, LogLevel.INFO, "Loading LabPics dataset...")
		data_dir	= Path("/mnt/raid1/dataset/LabPicsV1")
	
		train_dataset	= LabPicsDataset(data_dir, split="Train")
		train_loader	= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)  # drop_last handles variable batch sizes
		val_dataset	= LabPicsDataset(data_dir, split="Test")
		val_loader	= DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
	elif "spread" in dataset_name.lower():
		dbgprint(main, LogLevel.INFO, "Loading Spread dataset...")
		data_dir	= Path("/mnt/raid1/dataset/spread")
		
		train_dataset	= SpreadDataset(data_dir,   split="train")
		train_loader	= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
		
		val_dataset	= SpreadDataset(data_dir,   split="val")
		val_loader	= DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
		
		test_dataset	= SpreadDataset(data_dir,   split="test")
		test_loader	= DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
		
		dbgprint(dataloader, LogLevel.INFO, f"Train: {len(train_dataset)} - Val: {len(val_dataset)} - Test: {len(test_dataset)}")
	else:
		raise Exception(f"Unknown dataset: {dataset_name}")
	
	# Load model
	'''	
	sam2_checkpoint	= "sam2_hiera_small.pt"					# path to model weight
	model_cfg	= "sam2_hiera_s.yaml"					# model config
	
	sam2_checkpoint	= "sam2_hiera_large.pt"
	model_cfg	= "sam2_hiera_l.yaml"
	'''	
	
	sam2_model	= build_sam2(model_cfg, sam2_checkpoint, device="cuda")	# load model
	predictor	= SAM2ImagePredictor(sam2_model)

	# Magic
	if use_wandb:
		wandb.watch(predictor.model, log_freq=100)
	
	# Set training parameters
	
	predictor.model.sam_mask_decoder.train(True)				# enable training of mask decoder
	predictor.model.sam_prompt_encoder.train(True)				# enable training of prompt encoder
	predictor.model.image_encoder.train(True)				# enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
	
	optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
	scaler    = torch.cuda.amp.GradScaler() # mixed precision
	
	# Training loop
	
	best_loss = float("inf")
	best_iou  = float("-inf")
	mean_iou  = 0
	
	for epoch in range(num_epochs):  # Example: 100 epochs
		for itr, (image, mask, input_point) in enumerate(train_loader):
			with torch.cuda.amp.autocast():							# cast to mix precision
				# ... (training code remains largely the same, but use data from the loader)
	
				dbgprint(train, LogLevel.TRACE, f'{type(image) = } {type(mask) = } {type(input_point) = }')
				dbgprint(train, LogLevel.TRACE, f'{len(image)  = } {len(mask)  = } {len(input_point) = }')
				input_point = torch.tensor(np.array(input_point)).cuda().float()
				input_label = torch.ones(input_point.shape[0], 1).cuda().float() # create labels
	
				if isinstance(image, list):
					if len(image)==0:
						continue					# ignore empty batches
				if isinstance(mask, list):
					if len(mask)==0:
						continue					# ignore empty batches
				if isinstance(mask, torch.Tensor):
					if mask.shape[0]==0:
						continue					# ignore empty batches
	
				prd_masks, prd_scores 		= sam2_predict(predictor, image, mask, input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
				loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(mask, prd_masks, prd_scores, score_loss_weight=0.05)

				if use_wandb:
					wandb.log({"loss": loss, "seg_loss": seg_loss, "score_loss": score_loss, "iou": iou.mean().item(), "mean_iou": mean_iou, "epoch": epoch, "itr": itr, "best_iou": best_iou, "best_loss": best_loss})
					if itr == 0 and False:
						#wandb_images = wandb.Image(image, caption="Validation 1st batch")
						#wandb.log({"examples": images})
						dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'5. - {type(image) = } - {type(mask) = } - {type(prd_masks) = } - {type(prd_scores) = }')
						dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'6. - {len(image)  = } - {len(mask)  = } - {prd_masks.shape = } - {prd_scores.shape = }')
		
						table = wandb.Table(columns=["Image/Pred/GT"])
		
						class_labels = {1: "class1", 2: "class2"}
						for img, msk, prd_mask, prd_score in zip(image, mask, prd_masks, prd_scores):
							gt_msk   = torch.tensor(np.array(msk).astype(np.float32)).detach().cpu().numpy()
							prd_msk  = torch.sigmoid(prd_mask[:, 0]).detach().cpu().numpy()
							dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'7. - {gt_msk.shape = } - {prd_mask.shape = } - {prd_score.shape = }')
							dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'8. - {gt_msk = }')
							dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'9. - {prd_msk = }')
							dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'0. - {prd_score = }')
		
							mask_img = wandb.Image(img, masks={
												"prediction":	{"mask_data": prd_msk, "class_labels": class_labels},
												"ground_truth":	{"mask_data": gt_msk,  "class_labels": class_labels}
											})
							table.add_data(img)#, prd_score[:, 0])
						wandb.log({"Table": table})
	
				# apply back propogation
	
				predictor.model.zero_grad()					# empty gradient
				scaler.scale(loss).backward()					# Backpropagate
				scaler.step(optimizer)
				scaler.update()							# Mix precision
	
				# Display results
	
				mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
				dbgprint(train, LogLevel.INFO, f"step: {itr} - accuracy (IOU): {mean_iou:.2f} - loss: {loss:.2f} - seg_loss: {seg_loss:.2f} - score_loss: {score_loss:.2f} - best_loss: {best_loss:.2f}")
				if itr % 10 == 0:
					if loss < best_loss:
						# save the model
						best_loss = loss
						model_str = f"{sam2_checkpoint.replace('.pt','')}-{dataset_name}-training-epoch-{epoch}-step-{itr}-iou-{mean_iou:.2f}-best-loss-{loss:.2f}-segloss-{seg_loss:.2f}-scoreloss-{score_loss:.2f}.pth"
						dbgprint(train, LogLevel.INFO, f"Saving model: {model_str}")
						torch.save(predictor.model.state_dict(), model_str);
	
		avg_val_loss, avg_val_iou, avg_val_score_loss, avg_val_seg_loss, num_batches = validate(predictor, val_loader)
		dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'Num batches: {num_batches}, Loss: {avg_val_loss:.4f}, IOU: {avg_val_iou:.4f}, Score: {avg_val_score_loss:.4f}, Seg: {avg_val_seg_loss:.4f}')


		if use_wandb:
			wandb.log({"val_loss": avg_val_loss, "val_iou": avg_val_iou, "val_seg_loss": avg_val_seg_loss, "val_score_loss": avg_val_score_loss, "epoch": epoch, "best_iou": best_iou, "best_loss": best_loss, "mean_iou": mean_iou, "num_batches": num_batches})


		if avg_val_loss < best_loss or avg_val_iou > best_iou:
			best_loss = avg_val_loss
			best_iou  = avg_val_iou
			model_str = f"{sam2_checkpoint.replace('.pt','')}-{dataset_name}-validation-epoch-{epoch}-iou-{avg_val_iou:.2f}-best-loss-{avg_val_loss:.2f}-segloss-{avg_val_seg_loss:.2f}-scoreloss-{avg_val_score_loss:.2f}.pth"
			dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f"Saving model: {model_str}")
			torch.save(predictor.model.state_dict(), model_str);
