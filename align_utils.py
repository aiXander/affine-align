import numpy as np
import cv2
from typing import Tuple, Optional

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1]."""
    img_float = img.astype(np.float32)
    img_float = img_float - np.min(img_float)
    img_float = img_float / np.max(img_float)
    return img_float

def resize_maintain_aspect(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Resize image maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img
        scale = 1.0
        
    return resized, scale

def compute_center_of_mass(img: np.ndarray, threshold: int) -> Tuple[float, float]:
    """Compute center of mass for pixels above threshold."""
    # Apply threshold
    thresholded = np.where(img > threshold, img, 0)
    
    # Find non-zero pixels
    y_coords, x_coords = np.where(thresholded > 0)
    if len(x_coords) == 0:
        # If no pixels above threshold, return image center
        h, w = img.shape[:2]
        return w/2, h/2
    
    # Compute center of mass
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return center_x, center_y

def convert_rgba_to_rgb(img, background_color=0.0):
    """
    Convert RGBA image to RGB by multiplying RGB values with alpha channel
    and making transparent pixels equal to background_color
    """
    if len(img.shape) == 3 and img.shape[2] == 4:  # Has alpha channel
        # Extract RGB and alpha channels
        rgb = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3].astype(np.float32) / 255.0  # Normalize alpha to 0-1
        
        # Alpha blend with background color
        # Formula: result = alpha * foreground + (1 - alpha) * background
        background = np.full_like(rgb, background_color)
        result = alpha[:, :, np.newaxis] * rgb + (1 - alpha[:, :, np.newaxis]) * background
        
        # Convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    else:
        # Already RGB or grayscale, return as is
        return img.astype(np.uint8)