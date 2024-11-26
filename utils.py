import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

def create_model(model_size="small", checkpoint=None):
	# Load a pretrained model

	sam2_checkpoint, model_cfg = set_model_paths(model_size)

	sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
	
	# Build net and load weights
	predictor = SAM2ImagePredictor(sam2_model)
	if checkpoint is not None:
		predictor.model.load_state_dict(torch.load(checkpoint))

	return predictor
