import torch

def extract_mask_features(masks: torch.Tensor):
    """
    Extracts features from a batch of binary segmentation masks.

    Args:
        masks: A 4D tensor of shape (B, C, H, W) representing binary masks
               where 1 indicates the presence of an object (tree) and 0 otherwise.

    Returns:
        A dictionary containing lists of extracted features for each mask in the batch:
            - 'bboxes': List of bounding boxes, each as [x_min, y_min, x_max, y_max].
            - 'num_white_pixels': List of the number of white pixels in each mask.
            - 'widths': List of bounding box widths.
            - 'heights': List of bounding box heights.
            - 'diagonals': List of bounding box diagonals.
            - 'roundness_indices': List of roundness indices for each mask.
    """
    B, C, H, W = masks.shape
    features = {
        'bboxes': [],
        'num_white_pixels': [],
        'widths': [],
        'heights': [],
        'diagonals': [],
        'roundness_indices': []
    }

    for b in range(B):
        for c in range(C):
            mask = masks[b, c]
            white_pixels = torch.where(mask == 1)

            if white_pixels[0].numel() == 0:
                # Handle empty mask
                features['bboxes'].append(None)
                features['num_white_pixels'].append(0)
                features['widths'].append(0)
                features['heights'].append(0)
                features['diagonals'].append(0)
                features['roundness_indices'].append(0)
                continue

            y_min = white_pixels[0].min()
            x_min = white_pixels[1].min()
            y_max = white_pixels[0].max()
            x_max = white_pixels[1].max()

            bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            diagonal = torch.sqrt(width**2 + height**2)
            num_white = mask.sum()

            # Roundness index (area / area of bounding box's inscribed ellipse)
            # Assuming the white content fits roughly within the bounding box
            ellipse_area = torch.pi * (width / 2) * (height / 2)
            roundness = num_white / ellipse_area if ellipse_area > 0 else torch.tensor(0.0, device=masks.device)

            features['bboxes'].append(bbox)
            features['num_white_pixels'].append(num_white.item())
            features['widths'].append(width.item())
            features['heights'].append(height.item())
            features['diagonals'].append(diagonal.item())
            features['roundness_indices'].append(roundness.item())

    return features

def mask_cost_function(features: dict,
                       bbox_weight: float,
                       wh_weight: float,
                       white_pixels_weight: float,
                       diagonal_weight: float,
                       roundness_weight: float,
                       target_bboxes: list):
    """
    Calculates a cost based on the extracted features and target bounding box positions.

    Args:
        features: A dictionary of extracted features from `extract_mask_features`.
        bbox_weight: Weight for the bounding box position difference.
        wh_weight: Weight for the bounding box width and height difference.
        white_pixels_weight: Weight for the number of white pixels difference.
        diagonal_weight: Weight for the bounding box diagonal difference.
        roundness_weight: Weight for the roundness index difference.
        target_bboxes: A list of target bounding boxes (x_min, y_min, x_max, y_max).
                       Should have the same length as the number of masks.

    Returns:
        A tensor representing the cost for each mask.
    """
    num_masks = len(features['bboxes'])
    costs = torch.zeros(num_masks, dtype=torch.float32, device=next(iter(features.values())).device)

    for i in range(num_masks):
        if features['bboxes'][i] is None:
            # High cost for empty masks as they don't represent an instance
            costs[i] = 1e6
            continue

        bbox = torch.tensor(features['bboxes'][i], dtype=torch.float32, device=costs.device)
        target_bbox = torch.tensor(target_bboxes[i], dtype=torch.float32, device=costs.device)

        # Bounding Box Position Cost (e.g., L1 distance of corners)
        bbox_cost = torch.abs(bbox - target_bbox).sum() * bbox_weight

        # Width and Height Cost
        width_cost = torch.abs(features['widths'][i] - (target_bbox[2] - target_bbox[0] + 1)) * wh_weight
        height_cost = torch.abs(features['heights'][i] - (target_bbox[3] - target_bbox[1] + 1)) * wh_weight

        # Number of White Pixels Cost (you might want to normalize this)
        white_pixels_cost = torch.abs(features['num_white_pixels'][i] - ( (target_bbox[2] - target_bbox[0] + 1) * (target_bbox[3] - target_bbox[1] + 1) * 0.7 )) * white_pixels_weight # Assuming target mask is ~70% filled

        # Diagonal Cost
        target_width = target_bbox[2] - target_bbox[0] + 1
        target_height = target_bbox[3] - target_bbox[1] + 1
        target_diagonal = torch.sqrt(target_width**2 + target_height**2)
        diagonal_cost = torch.abs(features['diagonals'][i] - target_diagonal) * diagonal_weight

        # Roundness Cost (you might want to normalize this to be between 0 and 1)
        roundness_cost = torch.abs(features['roundness_indices'][i] - 1.0) * roundness_weight # Assuming a perfect circle has roundness ~ 1

        costs[i] = bbox_cost + width_cost + height_cost + white_pixels_cost + diagonal_cost + roundness_cost

    return costs

if __name__ == '__main__':
    # Example Usage
    batch_size = 2
    num_channels = 1
    height = 32
    width = 32

    # Create some dummy masks on GPU
    dummy_masks = torch.zeros((batch_size, num_channels, height, width), dtype=torch.float32).cuda()

    # Example mask 1 (a rectangle)
    dummy_masks[0, 0, 10:20, 5:25] = 1

    # Example mask 2 (a more circular shape)
    rr, cc = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_y, center_x = 16, 16
    radius = 8
    circular_mask = (rr - center_y)**2 + (cc - center_x)**2 <= radius**2
    dummy_masks[1, 0] = circular_mask.float().cuda()

    # Extract features
    extracted_features = extract_mask_features(dummy_masks)
    print("Extracted Features:", extracted_features)

    # Define weights for the cost function
    bbox_weight = 10.0
    wh_weight = 5.0
    white_pixels_weight = 2.0
    diagonal_weight = 0.5
    roundness_weight = 0.1

    # Define target bounding boxes (for demonstration, let's use the ground truth of the generated masks)
    target_bboxes = [[5, 10, 24, 19], [16 - radius, 16 - radius, 16 + radius, 16 + radius]]

    # Calculate the cost
    costs = mask_cost_function(extracted_features, bbox_weight, wh_weight,
                               white_pixels_weight, diagonal_weight, roundness_weight,
                               target_bboxes)
    print("Costs:", costs)
