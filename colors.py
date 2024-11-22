#!/usr/bin/env python3

from collections import defaultdict
import random

# Define a list of popular RGB colors
popular_colors = [
    {"name": "Black", "rgb": [0, 0, 0]},
    {"name": "Red", "rgb": [255, 0, 0]},
    {"name": "Green", "rgb": [0, 255, 0]},
    {"name": "Blue", "rgb": [0, 0, 255]},
    {"name": "Yellow", "rgb": [255, 255, 0]},
    {"name": "Cyan", "rgb": [0, 255, 255]},
    {"name": "Magenta", "rgb": [255, 0, 255]},
    {"name": "White", "rgb": [255, 255, 255]},
    {"name": "Gray", "rgb": [128, 128, 128]},
    {"name": "Orange", "rgb": [255, 165, 0]},
    {"name": "Purple", "rgb": [128, 0, 128]},
    {"name": "Pink", "rgb": [255, 192, 203]},
    {"name": "Brown", "rgb": [165, 42, 42]},
    {"name": "Lime", "rgb": [0, 255, 0]},
    {"name": "Teal", "rgb": [0, 128, 128]},
    {"name": "Maroon", "rgb": [128, 0, 0]},
    {"name": "Olive", "rgb": [107, 142, 35]},
    {"name": "Navy", "rgb": [0, 0, 128]},
    {"name": "Ivory", "rgb": [255, 255, 240]},
    {"name": "Beige", "rgb": [245, 245, 220]},
    {"name": "Gold", "rgb": [255, 215, 0]},
    {"name": "Silver", "rgb": [192, 192, 192]},
    {"name": "Bronze", "rgb": [205, 127, 50]},
    {"name": "Turquoise", "rgb": [64, 224, 208]},
    {"name": "Violet", "rgb": [238, 130, 238]},
    {"name": "Lavender", "rgb": [230, 230, 250]},
    {"name": "Coral", "rgb": [255, 127, 80]},
    {"name": "Azure", "rgb": [240, 255, 255]},
    {"name": "Khaki", "rgb": [240, 230, 140]},
    {"name": "Salmon", "rgb": [250, 128, 114]},
    {"name": "Sky Blue", "rgb": [135, 206, 235]},
    {"name": "Chartreuse", "rgb": [127, 255, 0]},
    {"name": "Dark Slate Gray", "rgb": [47, 79, 79]},
    {"name": "Slate Gray", "rgb": [112, 128, 144]},
    {"name": "Light Slate Gray", "rgb": [119, 136, 153]},
    {"name": "Light Steel Blue", "rgb": [176, 196, 222]},
    {"name": "Powder Blue", "rgb": [176, 224, 230]},
    {"name": "Dark Turquoise", "rgb": [0, 206, 209]},
    {"name": "Deep Sky Blue", "rgb": [0, 191, 255]},
    {"name": "Light Sky Blue", "rgb": [135, 206, 250]},
    {"name": "Light Blue", "rgb": [173, 216, 230]},
    {"name": "Dark Slate Blue", "rgb": [72, 61, 139]},
    {"name": "Slate Blue", "rgb": [106, 90, 205]},
    {"name": "Medium Slate Blue", "rgb": [123, 104, 238]},
    {"name": "Medium Blue", "rgb": [0, 0, 205]},
    {"name": "Royal Blue", "rgb": [65, 105, 225]},
    {"name": "Steel Blue", "rgb": [70, 130, 180]},
    {"name": "Dodger Blue", "rgb": [30, 144, 255]},
    {"name": "Cornflower Blue", "rgb": [100, 149, 237]},
    {"name": "Deep Sky Blue", "rgb": [0, 191, 255]},
    {"name": "Sky Blue", "rgb": [135, 206, 235]},
    {"name": "Light Sea Green", "rgb": [32, 178, 170]},
    {"name": "Honeydew", "rgb": [240, 255, 240]},
    {"name": "Mint Cream", "rgb": [245, 255, 250]},
    {"name": "Alice Blue", "rgb": [240, 248, 255]},
    {"name": "Ghost White", "rgb": [248, 248, 255]},
    {"name": "White Smoke", "rgb": [245, 245, 245]},
    {"name": "Seashell", "rgb": [255, 245, 238]},
    {"name": "Beige", "rgb": [245, 245, 220]},
    {"name": "Old Lace", "rgb": [253, 245, 230]},
    {"name": "Wheat", "rgb": [245, 222, 179]},
    {"name": "Sandy Brown", "rgb": [244, 164, 96]},
    {"name": "Tomato", "rgb": [255, 99, 71]},
    {"name": "Orange Red", "rgb": [255, 69, 0]},
    {"name": "Dark Orange", "rgb": [255, 140, 0]},
    {"name": "Light Salmon", "rgb": [255, 160, 122]},
    {"name": "Coral", "rgb": [255, 127, 80]},
    {"name": "Dark Salmon", "rgb": [233, 150, 122]},
    {"name": "Light Coral", "rgb": [240, 128, 128]},
    {"name": "Indian Red", "rgb": [205, 92, 92]},
    {"name": "Hot Pink", "rgb": [255, 105, 180]},
    {"name": "Deep Pink", "rgb": [255, 20, 147]},
    {"name": "Medium Violet Red", "rgb": [199, 21, 133]},
    {"name": "Pale Violet Red", "rgb": [219, 112, 147]},
    {"name": "Cyan", "rgb": [0, 255, 255]},
    {"name": "Light Cyan", "rgb": [224, 255, 255]},
    {"name": "Cadet Blue", "rgb": [95, 158, 160]},
]

# Function to get RGB by color name
def get_rgb_by_name(color_name):
    for color in popular_colors:
        if color["name"].lower() == color_name.lower():  # Case-insensitive match
            return color["rgb"]
    return None  # Return None if name not found

# Function to get color name by RGB
def get_name_by_rgb(rgb_value):
    for color in popular_colors:
        if color["rgb"] == rgb_value:
            return color["name"]
    return None  # Return None if RGB not found

def get_rgb_by_idx(idx):
	return popular_colors[idx]["rgb"]

def test_colors():
	# Example usage
	print(get_rgb_by_name("Coral"))			# Output: [255, 127, 80]
	print(get_name_by_rgb([233, 150, 122]))		# Output: "Dark Salmon"
	print(get_rgb_by_name("asdf"))			# Output: None (not found)
	print(get_name_by_rgb([251, 251, 251]))		# Output: None (not found)
	for c in popular_colors:
		print(c["name"], c["rgb"])

if __name__ == "__main__":
	test_colors()

