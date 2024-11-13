#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch

import pdb
import cv2

from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Dataset class
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

        #print(f'{type(Img)} {type(mask)} {type(point)}')
        return Img, mask, point

#def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor]]):
def collate_fn(data):
    print(f'collate_fn() 1. - {type(data)    = } {len(data)    = }')
    print(f'collate_fn() 2. - {type(data[0]) = } {len(data[0]) = }')
    print(f'collate_fn() 3. - {type(data[1]) = } {len(data[1]) = }')
    print(f'collate_fn() 4. - {type(data[2]) = } {len(data[2]) = }')
    imgs, masks, points = zip(*data)
    print(f'collate_fn() 5. - {type(imgs)    = } {len(imgs)    = }')
    print(f'collate_fn() 6. - {type(masks)   = } {len(masks)   = }')
    print(f'collate_fn() 7. - {type(points)  = } {len(points)  = }')
    return list(imgs), list(masks), list(points)
    #return torch.stack(imgs), torch.stack(masks), torch.stack(points)


# Set number of threads globally
NUM_WORKERS = 1
torch.set_num_threads(NUM_WORKERS)
_breakpoints = {}

def reset_breakpoints(disabled=[]):
    global _breakpoints
    _breakpoints = dict((x, False) for x in disabled)

def set_breakpoint(tag, condition=True):
    if tag not in _breakpoints:
        _breakpoints[tag] = True
        if condition:
            pdb.set_trace()
    else:
        if _breakpoints[tag] and condition:
            pdb.set_trace()

# Use with:
# set_breakpoint('mycode0')
# set_breakpoint('mycode1', x == 4)






# Data Loaders
data_dir = "/mnt/raid1/dataset/LabPicsV1/"
batch_size = 3
train_dataset = LabPicsDataset(data_dir, split="Train")
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, collate_fn=collate_fn)  # drop_last handles variable batch sizes
val_dataset   = LabPicsDataset(data_dir, split="Test")
val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, drop_last=True, collate_fn=collate_fn)

# Load model

sam2_checkpoint = "sam2_hiera_small.pt" # path to model weight
model_cfg = "sam2_hiera_s.yaml" #  model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler    = torch.cuda.amp.GradScaler() # mixed precision

# Training loop

best_loss = float("inf")

# Validation function
def validate(predictor, val_loader):
    total_loss = 0
    total_iou = 0
    count = 0
    predictor.model.eval()
    with torch.no_grad():
        for image, mask, input_point in val_loader:
            #image = torch.tensor(np.array(image)).cuda().permute(0, 3, 1, 2).float() / 255.0
            input_point = torch.tensor(np.array(input_point)).cuda().float()

            print(f'validate() 1. - {type(image)    = } - {type(mask)     = } - {type(input_point) = }')
            print(f'validate() 2. - {type(image)    = } - {len(image)     = }')
            print(f'validate() 3. - {type(image[0]) = } - {image[0].shape = } - {image[0].dtype = }')
            print(f'validate() 4. - {type(mask[0])  = } - {mask[0].shape  = } - {mask[0].dtype = }')

###########validate() 5. - low_res_masks.shape = torch.Size([3, 3, 256, 256]) - len(predictor._orig_hw) = 3
##########            predictor.set_image_batch(image)				# apply SAM image encoder to the image
##########            # predictor.get_image_embedding()
##########            # prompt encoding
##########
##########            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
##########            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
##########
##########            # mask decoder
##########
##########            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
##########            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"], image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
##########            print(f'train() 5. - {low_res_masks.shape = } - {len(predictor._orig_hw) = }')
##########            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution
###########validate() 5. - low_res_masks.shape = torch.Size([3, 3, 256, 256]) - len(predictor._orig_hw) = 3


            predictor.set_image_batch(image)
            # predictor.get_image_embedding()
            # prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(point_coords=input_point, point_labels=torch.ones(input_point.shape[0], 1).cuda(), box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

            # mask decoder

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],
                                                                             image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                                                                             sparse_prompt_embeddings=sparse_embeddings,
                                                                             dense_prompt_embeddings=dense_embeddings,
                                                                             multimask_output=True,
                                                                             repeat_image=False,
                                                                             high_res_features=high_res_features)


            print(f'validate() 5. - {low_res_masks.shape = } - {len(predictor._orig_hw) = }', flush=True)
            #set_breakpoint('validate0')
            #prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw)[:, 0]
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw)[-1]


            gt_mask = torch.tensor(np.array(mask).astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks)

            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-7) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-7)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            total_loss += loss.item()
            total_iou += iou.mean().item()
            count += 1

    return total_loss / count, total_iou / count


#val_loss, val_iou = validate(predictor, val_loader)
#print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation IOU: {val_iou:.4f}")


def sam2_predict(predictor, image, mask, input_point, input_label, box=None, mask_logits=None, normalize_coords=True):
	print(f'sam2_predict()() 1. - {type(image)    = } - {type(mask)     = } - {type(input_point) = }')
	print(f'sam2_predict()() 2. - {type(image)    = } - {len(image)     = }')
	print(f'sam2_predict()() 3. - {type(image[0]) = } - {image[0].shape = } - {image[0].dtype = }')
	print(f'sam2_predict()() 4. - {type(mask[0])  = } - {mask[0].shape  = } - {mask[0].dtype = }')

	#train() 1. - type(image)    = <class 'list'> - type(mask)     = <class 'list'> - type(input_point) = <class 'torch.Tensor'>
	#train() 2. - type(image)    = <class 'list'> - len(image)     = 3
	#train() 3. - type(image[0]) = <class 'numpy.ndarray'> - image[0].shape = (1024, 1024, 3) - image[0].dtype = dtype('uint8')
	#train() 4. - type(mask[0])  = <class 'numpy.ndarray'> - mask[0].shape  = (1024, 1024) - mask[0].dtype = dtype('uint8')
	#train() 5. - low_res_masks.shape = torch.Size([3, 3, 256, 256]) - len(predictor._orig_hw) = 3

	#validate() 1. - type(image)    = <class 'list'> - type(mask)     = <class 'list'> - type(input_point) = <class 'torch.Tensor'>
	#validate() 2. - type(image)    = <class 'list'> - len(image)     = 3
	#validate() 3. - type(image[0]) = <class 'numpy.ndarray'> - image[0].shape = (1024, 1024, 3) - image[0].dtype = dtype('uint8')
	#validate() 4. - type(mask[0])  = <class 'numpy.ndarray'> - mask[0].shape  = (1024, 1024) - mask[0].dtype = dtype('uint8')
	#validate() 5. - low_res_masks.shape = torch.Size([3, 3, 256, 256]) - len(predictor._orig_hw) = 3

	predictor.set_image_batch(image)				# apply SAM image encoder to the image
	# predictor.get_image_embedding()
	# prompt encoding

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

	print(f'sam2_predict() 5. - {low_res_masks.shape = } - {len(predictor._orig_hw) = }')
	# Upscale the masks to the original image resolution
	prd_masks					= predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

	return prd_masks, prd_scores


# Training loop
best_loss = float("inf")
for epoch in range(100):  # Example: 100 epochs
    for itr, (image, mask, input_point) in enumerate(train_loader):
        with torch.cuda.amp.autocast():							# cast to mix precision
            # ... (training code remains largely the same, but use data from the loader)

            #print(f'{type(image) = } {type(mask) = } {type(input_point) = }')
            #print(f'{len(image)  = } {len(mask)  =  } {len(input_point) = }')
            input_point = torch.tensor(np.array(input_point)).cuda().float()
            input_label = torch.ones(input_point.shape[0], 1).cuda().float() # create labels

            if isinstance(image, list):
                if len(image)==0:
                    continue							# ignore empty batches
            if isinstance(mask, list):
                if len(mask)==0:
                    continue							# ignore empty batches
            if isinstance(mask, torch.Tensor):
                if mask.shape[0]==0:
                    continue							# ignore empty batches

            ### predictor.set_image_batch(image)				# apply SAM image encoder to the image
            ### # predictor.get_image_embedding()
            ### # prompt encoding

            ### mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            ### sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

            ### # mask decoder

            ### high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            ### low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"], image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
            ### print(f'train() 5. - {low_res_masks.shape = } - {len(predictor._orig_hw) = }')
            ### prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution


            prd_masks, prd_scores, _, _ = sam2_predict(predictor, image, mask, input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            # Segmentaion Loss caclulation

            gt_mask  = torch.tensor(np.array(mask).astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])		# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss*0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            # Display results

            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"step: {itr} - accuracy (IOU): {mean_iou:.2f} - loss: {loss:.2f} - seg_loss: {seg_loss:.2f} - score_loss: {score_loss:.2f} - best_loss: {best_loss:.2f}")
            if itr % 10 == 0:
                if loss < best_loss:
                    # save the model
                    best_loss = loss
                    model_str = f"{sam2_checkpoint.replace('.pt','')}-step-{itr}-acc-{mean_iou:.2f}-best-loss-{loss:.2f}-segloss-{seg_loss:.2f}-scoreloss-{score_loss:.2f}.pth"
                    torch.save(predictor.model.state_dict(), model_str);

    val_loss, val_iou = validate(predictor, val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation IOU: {val_iou:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        # ... (save model)
