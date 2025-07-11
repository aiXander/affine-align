import cv2
import numpy as np
# from scipy.optimize import differential_evolution  # Replaced with CMA-ES
import cma
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
from align_utils import *

class VisualAligner:
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.best_score = float('inf')
        self.best_params = None
        
    def preprocess_image(self, img: np.ndarray, bypass_processor: bool = False, invert_pixels: bool = False) -> np.ndarray:
        """Preprocess image to extract edge features."""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        if bypass_processor: # Skip Canny edge detection, just use the grayscale image
            processed = gray
        else:
            # Use Canny edge detection
            processed = cv2.Canny(gray, 50, 150)
            # Make the canny edges thicker:
            processed = cv2.dilate(processed, np.ones((2, 2), np.uint8), iterations=1)
        
        if invert_pixels:
            processed = 255 - processed
            
        return processed


    def perceptual_loss(self, img_a: np.ndarray, img_b: np.ndarray, blur_fraction: float = 0.007) -> float:
        """
        Compute perceptual similarity score between two images using Gaussian-blurred cross-correlation.
        
        Args:
            img_a: First image (numpy array)
            img_b: Second image (numpy array)
            blur_fraction: Fraction of image size to use as Gaussian blur sigma (default: 0.01)
        
        Returns:
            float: Perceptual loss value (lower is better, 0 is perfect match)
        """
        # Ensure same shape
        if img_a.shape != img_b.shape:
            return float('inf')
        
        # Convert to float and normalize to [0, 1]:
        img_a_float = normalize_image(img_a)
        img_b_float = normalize_image(img_b)
        
        # Calculate Gaussian blur sigma based on image size
        # Use the smaller dimension to ensure blur is appropriate for both dimensions
        min_dim = min(img_a.shape[0], img_a.shape[1])
        sigma = blur_fraction * min_dim
        
        # Apply Gaussian blur to both images
        img_a_blurred = gaussian_filter(img_a_float, sigma=sigma)
        img_b_blurred = gaussian_filter(img_b_float, sigma=sigma)

        img_a_blurred = normalize_image(img_a_blurred)
        img_b_blurred = normalize_image(img_b_blurred)
        
        # Compute cross-correlation
        # Using 'valid' mode to avoid edge artifacts
        cross_corr = correlate2d(img_a_blurred, img_b_blurred, mode='valid')

        # Get the maximum cross-correlation value (best alignment)
        max_cross_corr = np.max(cross_corr)
        
        # Normalize cross-correlation by the auto-correlation values
        # This makes the metric more robust to intensity variations
        auto_corr_a = correlate2d(img_a_blurred, img_a_blurred, mode='valid')
        auto_corr_b = correlate2d(img_b_blurred, img_b_blurred, mode='valid')
        
        max_auto_corr_a = np.max(auto_corr_a)
        max_auto_corr_b = np.max(auto_corr_b)
        
        # Normalized cross-correlation coefficient
        if max_auto_corr_a == 0 or max_auto_corr_b == 0:
            return float('inf')
        
        normalized_cross_corr = max_cross_corr / np.sqrt(max_auto_corr_a * max_auto_corr_b)
        
        # Convert to loss (higher correlation = lower loss)
        # Clamp to avoid numerical issues
        normalized_cross_corr = np.clip(normalized_cross_corr, -1.0, 1.0)
        
        # Return loss as (1 - correlation) so that perfect match gives 0 loss
        loss = 1.0 - normalized_cross_corr
        
        return float(loss)

    def apply_affine_transform(self, img: np.ndarray, params: np.ndarray, 
                             target_shape: Tuple[int, int]) -> np.ndarray:
        """Apply affine transformation to image with independent X/Y scaling."""
        tx, ty, scale_x, scale_y, rotation = params
        h, w = img.shape[:2]
        th, tw = target_shape
        
        # Create transformation matrix manually to support non-uniform scaling
        center_x, center_y = w / 2, h / 2
        cos_r = np.cos(np.radians(rotation))
        sin_r = np.sin(np.radians(rotation))
        
        # Build transformation matrix: Scale, then Rotate, then Translate
        # Combined rotation + scaling matrix
        M = np.array([
            [scale_x * cos_r, -scale_y * sin_r, 0],
            [scale_x * sin_r, scale_y * cos_r, 0]
        ], dtype=np.float32)
        
        # Apply translation relative to center
        M[0, 2] = tx + center_x - (center_x * scale_x * cos_r - center_y * scale_y * sin_r)
        M[1, 2] = ty + center_y - (center_x * scale_x * sin_r + center_y * scale_y * cos_r)
        
        # Apply transformation
        transformed = cv2.warpAffine(img, M, (tw, th), 
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        
        return transformed
    
    def create_padded_image(self, img: np.ndarray, padding_fraction: float) -> Tuple[np.ndarray, int, int]:
        """Create padded version of image and return padding offsets."""
        h, w = img.shape[:2]
        pad_h = int(h * padding_fraction)
        pad_w = int(w * padding_fraction)
        
        # Create padded canvas (black background)
        padded_h = h + 2 * pad_h
        padded_w = w + 2 * pad_w
        padded_img = np.zeros((padded_h, padded_w), dtype=img.dtype)
        
        # Place original image in center of padded canvas
        padded_img[pad_h:pad_h + h, pad_w:pad_w + w] = img
        
        return padded_img, pad_w, pad_h
    
    def initial_affine_guess(self, img_photo_padded: np.ndarray, img_drawing: np.ndarray, threshold: int) -> np.ndarray:
        """Optimize affine parameters by matching centers of mass."""
        print("    Optimizing affine parameters...")
        
        # Compute centers of mass
        photo_cx, photo_cy = compute_center_of_mass(img_photo_padded, threshold)
        drawing_cx, drawing_cy = compute_center_of_mass(img_drawing, threshold)
        
        # Calculate translation to align centers of mass
        # We need to account for the coordinate system difference
        tx = drawing_cx - photo_cx
        ty = drawing_cy - photo_cy
        
        # Create affine parameters: [tx, ty, scale=1, rotation=0]
        params = np.array([tx, ty, 1.0, 1.0, 0.0]) # Initialize scale_x and scale_y to 1
        
        print(f"    Computed translation: ({tx:.1f}, {ty:.1f})")
        
        return params
    
    def apply_final_transformation(self, original_photo: np.ndarray, affine_params: np.ndarray,
                                 target_shape: Tuple[int, int], pad_offset_x: int, pad_offset_y: int) -> np.ndarray:
        """Apply final transformation from original photo to target drawing dimensions."""
        tx, ty, scale_x, scale_y, rotation = affine_params
        h_orig, w_orig = original_photo.shape[:2]
        th, tw = target_shape
        
        # Create transformation matrix
        # First, we need to account for the padding offset in our transformation
        # The optimization was done in padded space, but we're applying to original photo
        
        # Adjust translation to account for padding
        adjusted_tx = tx + pad_offset_x
        adjusted_ty = ty + pad_offset_y
        
        # Build transformation matrix manually for more control
        cos_r = np.cos(np.radians(rotation))
        sin_r = np.sin(np.radians(rotation))
        
        # Create the transformation matrix
        M = np.array([
            [scale_x * cos_r, -scale_y * sin_r, adjusted_tx],
            [scale_x * sin_r, scale_y * cos_r, adjusted_ty]
        ], dtype=np.float32)
        
        # Apply transformation
        transformed = cv2.warpAffine(original_photo, M, (tw, th), 
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        
        return transformed
    
    def coarse_alignment(self, img_photo: np.ndarray, img_drawing: np.ndarray, padding_fraction: float = 0.5) -> Tuple[np.ndarray, float]:
        """Perform coarse alignment using center of mass matching with clean separation of concerns."""
        print("  Coarse alignment: using center of mass matching")
        
        # Threshold filtering - keep only pixels above 0.25*255
        threshold = int(0.25 * 255)
        print(f"    Using threshold: {threshold}")
        
        # Create padded version of photo for optimization
        padded_photo, pad_offset_x, pad_offset_y = self.create_padded_image(img_photo, padding_fraction)
        
        print(f"    Original photo shape: {img_photo.shape}")
        print(f"    Padded photo shape: {padded_photo.shape}")
        print(f"    Padding offsets: ({pad_offset_x}, {pad_offset_y})")
        print(f"    Drawing shape: {img_drawing.shape}")
        
        # Optimize affine parameters using padded_photo and img_drawing:
        affine_params_padded = self.initial_affine_guess(padded_photo, img_drawing, threshold)
        
        # Apply final transformation from original photo to target dimensions for debug
        h_d, w_d = img_drawing.shape[:2]
        warped = self.apply_final_transformation(img_photo, affine_params_padded, (h_d, w_d),
                                               pad_offset_x, pad_offset_y)
        
        # Save debug images
        cv2.imwrite("debug_edges_photo_warped_1.jpg", warped)
        
        # Compute score
        score = self.perceptual_loss(warped, img_drawing)
        
        # IMPORTANT: Convert parameters from padded space to original photo space
        # This ensures fine_alignment can use them directly with apply_affine_transform
        affine_params_original = affine_params_padded.copy()
        affine_params_original[0] += pad_offset_x  # Adjust tx for padding
        affine_params_original[1] += pad_offset_y  # Adjust ty for padding
        # scale and rotation remain the same
        
        print(f"    Padded space params: tx={affine_params_padded[0]:.1f}, ty={affine_params_padded[1]:.1f}, scale_x={affine_params_padded[2]:.1f}, scale_y={affine_params_padded[3]:.1f}, rotation={affine_params_padded[4]:.1f}")
        print(f"    Original space params: tx={affine_params_original[0]:.1f}, ty={affine_params_original[1]:.1f}, scale_x={affine_params_original[2]:.1f}, scale_y={affine_params_original[3]:.1f}, rotation={affine_params_original[4]:.1f}")
        
        return affine_params_original, score
    
    def fine_alignment(self, img_photo: np.ndarray, img_drawing: np.ndarray,
                      initial_params: np.ndarray) -> np.ndarray:
        """Fine-tune alignment using CMA-ES optimization with early stopping."""

        h_d, w_d = img_drawing.shape[:2]
        
        # Compute initial score to compare against
        initial_warped = self.apply_affine_transform(img_photo, initial_params, (h_d, w_d))
        initial_score = self.perceptual_loss(initial_warped, img_drawing)
        
        # Progress tracking
        iteration_count = 0
        best_score = initial_score
        best_params = initial_params.copy()
        best_iteration = 0 
        loss_history = []
        
        def objective(params):
            nonlocal iteration_count, best_score, best_params, best_iteration, loss_history
            iteration_count += 1
            
            warped = self.apply_affine_transform(img_photo, params, (h_d, w_d))
            score = self.perceptual_loss(warped, img_drawing)
            loss_history.append(score)
            
            # Track best parameters
            if score < best_score:
                best_score = score
                best_iteration = iteration_count
                best_params = params.copy()
            elif iteration_count % 20 == 0:
                print(f"    Iteration {iteration_count:04d}: score: {score:.6f} (best: {best_score:.6f} at iteration {best_iteration:04d})")
            
            return score
        
        # Define step sizes for CMA-ES based on image dimensions
        tx_sigma = w_d * 0.25
        ty_sigma = h_d * 0.25
        scale_sigma = 0.25
        rotation_sigma = 25
        
        # Initial step sizes (sigma) for each parameter
        initial_sigma = np.array([tx_sigma, ty_sigma, scale_sigma, scale_sigma, rotation_sigma]) # Adjusted sigma for scale_x and scale_y
        
        # Define bounds for CMA-ES (samples outside these bounds are rejected)
        tx_range = w_d * 0.5
        ty_range = h_d * 0.5
        scale_range = 0.5
        rotation_range = 90
        
        lower_bounds = np.array([
            initial_params[0] - tx_range,
            initial_params[1] - ty_range,
            initial_params[2] * (1 - scale_range),
            initial_params[3] * (1 - scale_range), # Lower bound for scale_x
            initial_params[4] - rotation_range
        ])
        
        upper_bounds = np.array([
            initial_params[0] + tx_range,
            initial_params[1] + ty_range,
            initial_params[2] * (1 + scale_range),
            initial_params[3] * (1 + scale_range), # Upper bound for scale_x
            initial_params[4] + rotation_range
        ])
        
        print(f"Affine parameters:")
        print(f"    Translation step sizes: ±{tx_sigma:.1f}px (±{100*tx_sigma/w_d:.1f}% of width), ±{ty_sigma:.1f}px (±{100*ty_sigma/h_d:.1f}% of height)")
        print(f"    Scale step size: ±{100*scale_sigma:.1f}%")
        print(f"    Rotation step size: ±{rotation_sigma:.1f}°")
        print(f"    Translation bounds: ±{tx_range:.1f}px, ±{ty_range:.1f}px")
        print(f"    Scale_X bounds: {initial_params[2] * (1 - scale_range):.3f} to {initial_params[2] * (1 + scale_range):.3f}")
        print(f"    Scale_Y bounds: {initial_params[3] * (1 - scale_range):.3f} to {initial_params[3] * (1 + scale_range):.3f}")
        print(f"    Rotation bounds: {initial_params[4] - rotation_range:.1f}° to {initial_params[4] + rotation_range:.1f}°")
        
        # CMA-ES options
        options = {
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'maxiter': 30,            # Maximum iterations
            'popsize': 15,            # Population size (similar to DE)
            'seed': 42,               # Random seed for reproducibility
            'verb_disp': 0,           # Reduce verbosity
            'verb_log': 0,
            'CMA_stds': initial_sigma.tolist(),  # Initial step sizes
            'tolx': 1e-4,             # Tolerance for parameter changes
            'tolfun': 1e-5            # Tolerance for function value changes
        }
        
        # Run CMA-ES optimization
        try:
            es = cma.CMAEvolutionStrategy(initial_params, 0.1, options)
            es.optimize(objective)
            
            # Get results
            result_params = es.result.xbest
            final_score = es.result.fbest
            n_iterations = es.result.evaluations
            
        except Exception as e:
            print(f"    CMA-ES optimization failed: {e}")
            print(f"    Falling back to initial parameters")
            result_params = initial_params
            final_score = initial_score
            n_iterations = 0
        
        print(f"  Final CMA-ES score: {final_score:.6f}, function evaluations: {n_iterations}")
        print(f"  Best score found: {best_score:.6f}")

        # Plot loss history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('CMA-ES Optimization Progress')
        plt.grid(True)
        plt.savefig("loss_history_cmaes.png")
        plt.close()
        
        # Early stopping: only use fine-tuned params if they're actually better
        improvement = initial_score - best_score
        improvement_threshold = initial_score * 0.001  # Require at least 0.1% improvement
        
        if improvement > improvement_threshold:
            print(f"  Fine alignment improved score by {improvement:.6f} ({100*improvement/initial_score:.2f}%)")
            print(f"  Using fine-tuned parameters")
            return best_params
        else:
            print(f"  Fine alignment did not improve significantly (improvement: {improvement:.6f})")
            print(f"  Reverting to coarse alignment parameters")
            return initial_params
    
    def align_images(self, photo_path: str, drawing_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """Main alignment function."""
        print("Starting image alignment process...")
        
        # Load images
        img_photo = cv2.imread(photo_path)
        img_drawing = cv2.imread(drawing_path)

        # Convert RGBA to RGB: (alpha = black background)
        img_photo = convert_rgba_to_rgb(img_photo)
        img_drawing = convert_rgba_to_rgb(img_drawing)
        
        if img_photo is None or img_drawing is None:
            raise ValueError("Could not load images")
        
        # Store original photo for final transformation
        original_photo = img_photo.copy()
        
        # Resize for processing
        img_photo_resized, photo_scale = resize_maintain_aspect(img_photo, self.max_size)
        img_drawing_resized, drawing_scale = resize_maintain_aspect(img_drawing, self.max_size)
        print(f"  Photo: {img_photo.shape} -> {img_photo_resized.shape}")
        print(f"  Drawing: {img_drawing.shape} -> {img_drawing_resized.shape}")
        
        # Preprocess to edge space
        # For photos: apply Canny edge detection
        edges_photo = self.preprocess_image(img_photo_resized, bypass_processor=False, invert_pixels=False)
        # For drawings: bypass Canny and invert pixels to match edge-like appearance
        edges_drawing = self.preprocess_image(img_drawing_resized, bypass_processor=True, invert_pixels=True)
        
        # Save preprocessed images for debugging
        cv2.imwrite("debug_edges_photo_base.jpg", edges_photo)
        cv2.imwrite("debug_edges_drawing.jpg", edges_drawing)
        cv2.imwrite("debug_edges_photo_zcompare.jpg", edges_drawing)
        
        # Coarse alignment
        coarse_params, coarse_score = self.coarse_alignment(edges_photo, edges_drawing)
        print(f"Coarse alignment score: {coarse_score:.6f}")
        
        final_params = self.fine_alignment(edges_photo, edges_drawing, coarse_params)
        
        # Save post-alignment warped image for comparison
        h_d, w_d = edges_drawing.shape[:2]
        post_alignment_warped = self.apply_affine_transform(edges_photo, final_params, (h_d, w_d))
        
        cv2.imwrite("debug_edges_photo_warped_2.jpg", post_alignment_warped)
        print("Post-fine-alignment warped photo saved as debug_post_fine_warped.jpg")
        
        # Scale parameters back to original resolution
        scale_ratio = 1 / drawing_scale
        final_params = final_params.copy()
        print(f"  Scaling parameters from resized to original resolution:")
        print(f"    Drawing scale: {drawing_scale:.3f}, scale ratio: {scale_ratio:.3f}")
        print(f"    Before scaling: tx={final_params[0]:.1f}, ty={final_params[1]:.1f}")
        final_params[0] *= scale_ratio  # tx
        final_params[1] *= scale_ratio  # ty
        print(f"    After scaling: tx={final_params[0]:.1f}, ty={final_params[1]:.1f}")
        # scale_x, scale_y and rotation remain the same
        
        # Apply to original photo
        h_orig, w_orig = img_drawing.shape[:2]
        print(f"    Target shape (original drawing): {h_orig}x{w_orig}")
        print(f"    Original photo shape: {original_photo.shape}")
        aligned_photo = self.apply_affine_transform(original_photo, final_params,
                                                   (h_orig, w_orig))
        
        # Handle output path and saving
        saved_path = None
        if output_path is not None:
            # Generate output path if not provided
            if output_path == "":
                base_name = os.path.splitext(os.path.basename(photo_path))[0]
                output_path = f"{base_name}_aligned.jpg"
            
            cv2.imwrite(output_path, aligned_photo)
            saved_path = output_path
            print(f"Aligned photo saved to: {output_path}")
        
        return aligned_photo, final_params, saved_path

if __name__ == "__main__":
    photo_path = "3_photo.jpeg"  # Replace with your photo path
    drawing_path = "3_drawing.jpeg"  # Replace with your drawing path
    
    aligner = VisualAligner()
    aligned_photo, params, saved_path = aligner.align_images(photo_path, drawing_path, "custom_aligned.jpg")   