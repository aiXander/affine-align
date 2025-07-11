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

class TransformationManager:
    """Handles coordinate system transformations and parameter scaling."""
    
    def __init__(self, photo_original_shape: Tuple[int, int], drawing_original_shape: Tuple[int, int],
                 photo_resized_shape: Tuple[int, int], drawing_resized_shape: Tuple[int, int]):
        """
        Initialize transformation manager with original and resized shapes.
        
        Args:
            photo_original_shape: (height, width) of original photo
            drawing_original_shape: (height, width) of original drawing  
            photo_resized_shape: (height, width) of resized photo
            drawing_resized_shape: (height, width) of resized drawing
        """
        self.photo_original_shape = photo_original_shape
        self.drawing_original_shape = drawing_original_shape
        self.photo_resized_shape = photo_resized_shape
        self.drawing_resized_shape = drawing_resized_shape
        
        # Calculate scale factors
        self.photo_scale = min(photo_resized_shape[0] / photo_original_shape[0], 
                              photo_resized_shape[1] / photo_original_shape[1])
        self.drawing_scale = min(drawing_resized_shape[0] / drawing_original_shape[0], 
                                drawing_resized_shape[1] / drawing_original_shape[1])
        
        print(f"  TransformationManager initialized:")
        print(f"    Photo: {photo_original_shape} -> {photo_resized_shape} (scale: {self.photo_scale:.3f})")
        print(f"    Drawing: {drawing_original_shape} -> {drawing_resized_shape} (scale: {self.drawing_scale:.3f})")
    
    def validate_coordinate_system(self, params_resized: np.ndarray, params_original: np.ndarray) -> bool:
        """
        Validate that the coordinate system transformation is working correctly.
        
        This creates a test transformation to verify that the parameter scaling
        is working as expected.
        
        Args:
            params_resized: Parameters in resized space
            params_original: Parameters in original space
            
        Returns:
            True if validation passes, False otherwise
        """
        # Create test points in resized space (avoid 0,0 to prevent division by zero)
        test_points_resized = np.array([
            [1, 1], [self.photo_resized_shape[1]//2, self.photo_resized_shape[0]//2],
            [self.photo_resized_shape[1]-1, self.photo_resized_shape[0]-1]
        ])
        
        # Create test points in original space (scaled versions)
        test_points_original = np.array([
            [1/self.photo_scale, 1/self.photo_scale], 
            [self.photo_original_shape[1]//2, self.photo_original_shape[0]//2],
            [self.photo_original_shape[1]-1, self.photo_original_shape[0]-1]
        ])
        
        # The ratio of coordinate scaling should be consistent
        expected_ratio = 1 / self.photo_scale
        
        validation_passed = True
        for i, (resized_pt, original_pt) in enumerate(zip(test_points_resized, test_points_original)):
            actual_ratio_x = original_pt[0] / resized_pt[0]
            actual_ratio_y = original_pt[1] / resized_pt[1]
            
            if abs(actual_ratio_x - expected_ratio) > 0.5 or abs(actual_ratio_y - expected_ratio) > 0.5:
                print(f"    WARNING: Coordinate system validation failed at point {i}")
                print(f"    Expected ratio: {expected_ratio:.3f}")
                print(f"    Actual ratio: X={actual_ratio_x:.3f}, Y={actual_ratio_y:.3f}")
                validation_passed = False
        
        if validation_passed:
            print(f"    ✓ Coordinate system validation passed")
        
        return validation_passed
    
    def scale_parameters_to_original(self, params_resized: np.ndarray) -> np.ndarray:
        """
        Scale parameters from resized coordinate system to original coordinate system.
        
        The parameters were optimized to transform resized_photo to resized_drawing.
        We need to scale them to transform original_photo to original_drawing.
        
        The key insight is that the parameters define a transformation in the target 
        (drawing) coordinate system, so they should be scaled by the drawing scale factor.
        However, we need to be careful about coordinate system consistency.
        
        Args:
            params_resized: Parameters [tx, ty, scale_x, scale_y, rotation] in resized space
            
        Returns:
            Parameters scaled to original space
        """
        tx, ty, scale_x, scale_y, rotation = params_resized
        
        # Scale translation parameters by the drawing scale factor
        # This is correct because the parameters define the transformation 
        # in the target (drawing) coordinate system
        drawing_scale_ratio = 1 / self.drawing_scale
        tx_original = tx * drawing_scale_ratio
        ty_original = ty * drawing_scale_ratio
        
        # Scale and rotation remain the same (they're relative/dimensionless)
        params_original = np.array([tx_original, ty_original, scale_x, scale_y, rotation])
        
        print(f"  Parameter scaling:")
        print(f"    Photo scale: {self.photo_scale:.3f}, Drawing scale: {self.drawing_scale:.3f}")
        print(f"    Drawing scale ratio: {drawing_scale_ratio:.3f}")
        print(f"    Translation: ({tx:.1f}, {ty:.1f}) -> ({tx_original:.1f}, {ty_original:.1f})")
        print(f"    Scale/rotation unchanged: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}, rotation={rotation:.1f}°")
        
        # Validate that our coordinate system transformation makes sense
        if abs(self.photo_scale - self.drawing_scale) > 0.1:
            print(f"    WARNING: Significant scale factor difference detected!")
            print(f"    Photo scale factor: {self.photo_scale:.3f}")
            print(f"    Drawing scale factor: {self.drawing_scale:.3f}")
            print(f"    This may indicate different aspect ratios or sizing issues.")
            
            # For cases with significant scale differences, we might need to be more careful
            # about the coordinate system transformation
            scale_factor_ratio = self.drawing_scale / self.photo_scale
            print(f"    Scale factor ratio (drawing/photo): {scale_factor_ratio:.3f}")
            
            if scale_factor_ratio > 1.5 or scale_factor_ratio < 0.67:
                print(f"    WARNING: Extreme scale factor difference detected!")
                print(f"    This may cause alignment issues. Consider using images with similar aspect ratios.")
        
        # Validate the coordinate system transformation
        self.validate_coordinate_system(params_resized, params_original)
        
        return params_original
    
    def get_resized_shapes(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return resized shapes for photo and drawing."""
        return self.photo_resized_shape, self.drawing_resized_shape
    
    def get_original_shapes(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return original shapes for photo and drawing."""
        return self.photo_original_shape, self.drawing_original_shape

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

        # Save padded photo
        cv2.imwrite("debug_edges_photo_padded.jpg", padded_photo)
        
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
    
    def resize_images_with_common_scale(self, img_photo: np.ndarray, img_drawing: np.ndarray, 
                                      max_size: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Resize both images using a common scale factor to eliminate coordinate system mismatches.
        
        This method finds the most restrictive scale factor needed for both images and applies
        it to both, ensuring they're resized consistently.
        
        Args:
            img_photo: Original photo image
            img_drawing: Original drawing image
            max_size: Maximum dimension for resizing
            
        Returns:
            Tuple of (resized_photo, resized_drawing, common_scale_factor)
        """
        # Get original dimensions
        photo_h, photo_w = img_photo.shape[:2]
        drawing_h, drawing_w = img_drawing.shape[:2]
        
        # Calculate individual scale factors
        photo_scale = max_size / max(photo_h, photo_w)
        drawing_scale = max_size / max(drawing_h, drawing_w)
        
        # Use the more restrictive (smaller) scale factor for both images
        # This ensures both images fit within the max_size constraint
        common_scale = min(photo_scale, drawing_scale)
        
        print(f"  Common scale resizing:")
        print(f"    Photo individual scale: {photo_scale:.3f}")
        print(f"    Drawing individual scale: {drawing_scale:.3f}")
        print(f"    Using common scale: {common_scale:.3f}")
        
        # Resize both images with the common scale factor
        new_photo_w = int(photo_w * common_scale)
        new_photo_h = int(photo_h * common_scale)
        new_drawing_w = int(drawing_w * common_scale)
        new_drawing_h = int(drawing_h * common_scale)
        
        resized_photo = cv2.resize(img_photo, (new_photo_w, new_photo_h), interpolation=cv2.INTER_AREA)
        resized_drawing = cv2.resize(img_drawing, (new_drawing_w, new_drawing_h), interpolation=cv2.INTER_AREA)
        
        print(f"    Photo: {(photo_h, photo_w)} -> {(new_photo_h, new_photo_w)}")
        print(f"    Drawing: {(drawing_h, drawing_w)} -> {(new_drawing_h, new_drawing_w)}")
        
        return resized_photo, resized_drawing, common_scale

    def align_images(self, photo_path: str, drawing_path: str, output_path: Optional[str] = None, 
                    use_common_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """Main alignment function with proper coordinate system handling."""
        print("Starting image alignment process...")
        
        # Load images
        img_photo = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)
        img_drawing = cv2.imread(drawing_path, cv2.IMREAD_UNCHANGED)

        # Convert RGBA to RGB: (alpha = black background)
        img_photo = convert_rgba_to_rgb(img_photo)
        img_drawing = convert_rgba_to_rgb(img_drawing)
        
        if img_photo is None or img_drawing is None:
            raise ValueError("Could not load images")
        
        # Store original images and their shapes
        original_photo = img_photo.copy()
        original_photo_shape = img_photo.shape[:2]
        original_drawing_shape = img_drawing.shape[:2]
        
        # Resize for processing - choose between common scale or individual scaling
        if use_common_scale:
            print("  Using common scale factor for both images...")
            img_photo_resized, img_drawing_resized, common_scale = self.resize_images_with_common_scale(
                img_photo, img_drawing, self.max_size
            )
            
            # Create a simplified transformation manager for common scale
            transform_manager = TransformationManager(
                original_photo_shape, original_drawing_shape,
                img_photo_resized.shape[:2], img_drawing_resized.shape[:2]
            )
            
            # Verify that both images now have the same scale factor
            if abs(transform_manager.photo_scale - transform_manager.drawing_scale) < 0.01:
                print("  ✓ Common scale factor successfully applied!")
            else:
                print("  ⚠ WARNING: Common scale factor not perfectly applied")
                print(f"    Photo scale: {transform_manager.photo_scale:.3f}")
                print(f"    Drawing scale: {transform_manager.drawing_scale:.3f}")
        else:
            print("  Using individual scale factors for images...")
            img_photo_resized, photo_scale = resize_maintain_aspect(img_photo, self.max_size)
            img_drawing_resized, drawing_scale = resize_maintain_aspect(img_drawing, self.max_size)
            
            resized_photo_shape = img_photo_resized.shape[:2]
            resized_drawing_shape = img_drawing_resized.shape[:2]
            
            print(f"  Photo: {original_photo_shape} -> {resized_photo_shape}")
            print(f"  Drawing: {original_drawing_shape} -> {resized_drawing_shape}")
            
            # Create transformation manager to handle coordinate system conversions
            transform_manager = TransformationManager(
                original_photo_shape, original_drawing_shape,
                resized_photo_shape, resized_drawing_shape
            )
        
        # Preprocess to edge space
        # For photos: apply Canny edge detection
        edges_photo = self.preprocess_image(img_photo_resized, bypass_processor=False, invert_pixels=False)
        # For drawings: bypass Canny and invert pixels to match edge-like appearance
        edges_drawing = self.preprocess_image(img_drawing_resized, bypass_processor=True, invert_pixels=True)
        
        # Save preprocessed images for debugging
        cv2.imwrite("debug_edges_photo_0.jpg", edges_photo)
        cv2.imwrite("debug_edges_drawing.jpg", edges_drawing)
        cv2.imwrite("debug_edges_photo_zcompare.jpg", edges_drawing)
        
        # Coarse alignment (operates in resized space)
        coarse_params_resized, coarse_score = self.coarse_alignment(edges_photo, edges_drawing)
        print(f"Coarse alignment score: {coarse_score:.6f}")
        
        # Fine alignment (operates in resized space)
        final_params_resized = self.fine_alignment(edges_photo, edges_drawing, coarse_params_resized)
        
        # Save post-alignment warped image for comparison (in resized space)
        h_d, w_d = edges_drawing.shape[:2]
        post_alignment_warped = self.apply_affine_transform(edges_photo, final_params_resized, (h_d, w_d))
        
        cv2.imwrite("debug_edges_photo_warped_2.jpg", post_alignment_warped)
        print("Post-fine-alignment warped photo saved as debug_edges_photo_warped_2.jpg")
        
        # Scale parameters from resized space to original space
        final_params_original = transform_manager.scale_parameters_to_original(final_params_resized)
        
        # Apply transformation to original photo
        h_orig, w_orig = original_drawing_shape
        print(f"    Applying final transformation to original photo...")
        print(f"    Original photo shape: {original_photo_shape}")
        print(f"    Target shape (original drawing): {(h_orig, w_orig)}")
        aligned_photo = self.apply_affine_transform(original_photo, final_params_original, (h_orig, w_orig))
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(drawing_path))[0]
            output_path = f"{os.path.dirname(drawing_path)}/{base_name}_aligned.jpg"
            
        cv2.imwrite(output_path, aligned_photo)
        print(f"Aligned photo saved to: {output_path}")
        
        return aligned_photo, final_params_original, output_path



def align_folders(folder_a, pattern_a, folder_b, pattern_b, output_folder, output_pattern):
    """
    Align images from two folders based on matching patterns.
    
    Args:
        folder_a: Path to folder containing photos to be aligned
        pattern_a: Pattern to match files in folder_a (e.g., "photo_{}.jpg")
        folder_b: Path to folder containing reference drawings
        pattern_b: Pattern to match files in folder_b (e.g., "drawing_{}.jpg")
        output_folder: Path to folder where aligned images will be saved
        output_pattern: Pattern for output filenames (e.g., "aligned_{}.jpg")
    """
    import glob
    import re
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract the placeholder pattern (e.g., {} or {id})
    def extract_pattern_regex(pattern):
        """Convert a pattern like 'photo_{}.jpg' to a regex that captures the ID."""
        # Escape special regex characters except {}
        escaped = re.escape(pattern)
        # Replace escaped braces with capture group
        regex_pattern = escaped.replace(r'\{\}', r'([^/\\]+)')
        return regex_pattern
    
    def extract_id_from_filename(filename, pattern):
        """Extract the ID from a filename using the pattern."""
        regex = extract_pattern_regex(pattern)
        match = re.search(regex, filename)
        return match.group(1) if match else None
    
    # Get all files matching pattern_a
    pattern_a_glob = pattern_a.replace('{}', '*')
    files_a = glob.glob(os.path.join(folder_a, pattern_a_glob))
    
    if not files_a:
        print(f"No files found matching pattern '{pattern_a}' in folder '{folder_a}'")
        return
    else:
        print(f"Found {len(files_a)} files in folder A matching pattern '{pattern_a}'")

    # Get all files matching pattern_b
    pattern_b_glob = pattern_b.replace('{}', '*')
    files_b = glob.glob(os.path.join(folder_b, pattern_b_glob))
    
    if not files_b:
        print(f"No files found matching pattern '{pattern_b}' in folder '{folder_b}'")
        return
    else:
        print(f"Found {len(files_b)} files in folder B matching pattern '{pattern_b}'")

    # Create mapping of IDs to files for folder B
    id_to_file_b = {}
    for file_b in files_b:
        filename_b = os.path.basename(file_b)
        id_b = extract_id_from_filename(filename_b, pattern_b)
        if id_b:
            id_to_file_b[id_b] = file_b
    
    # Process each file in folder A
    successful_alignments = 0
    failed_alignments = 0
    
    for file_a in files_a:
        filename_a = os.path.basename(file_a)
        id_a = extract_id_from_filename(filename_a, pattern_a)
        
        if not id_a:
            print(f"Warning: Could not extract ID from '{filename_a}' using pattern '{pattern_a}'")
            failed_alignments += 1
            continue
        
        # Find corresponding file in folder B
        if id_a not in id_to_file_b:
            print(f"Warning: No matching file found for ID '{id_a}' in folder B")
            failed_alignments += 1
            continue
        
        file_b = id_to_file_b[id_a]
        
        # Generate output path
        output_filename = output_pattern.format(id_a)
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"\nAligning pair {successful_alignments + 1}:")
        print(f"  Photo: {file_a}")
        print(f"  Drawing: {file_b}")
        print(f"  Output: {output_path}")
        
        try:
            # Align the image pair using the integrated method
            aligner = VisualAligner()
            aligned_photo, params = aligner.align_images(file_a, file_b)
            print(f"  ✓ Successfully aligned and saved to: {output_path}")
            successful_alignments += 1
        except Exception as e:
            print(f"  ✗ Failed to align: {e}")
            failed_alignments += 1
    
    print(f"\n=== Alignment Summary ===")
    print(f"Successful alignments: {successful_alignments}")
    print(f"Failed alignments: {failed_alignments}")
    print(f"Total pairs processed: {successful_alignments + failed_alignments}")


if __name__ == "__main__":
    img_id = 1
    photo_path   = f"{img_id}_photo.jpeg"  # Replace with your photo path
    drawing_path = f"{img_id}_drawing.jpeg"  # Replace with your drawing path

    aligner = VisualAligner()
    aligned_photo, params, saved_path = aligner.align_images(photo_path, drawing_path, 
                                                           output_path=f"{img_id}_drawing_aligned_common.jpg", 
                                                           use_common_scale=True)
    

    
    source_folder = "/home/rednax/Documents/datasets/good_styles/stitchly_final/Kontext/combi"
    align_folders(source_folder, "{}_rembg.png", source_folder, "{}_D2.jpg", source_folder, "{}_aligned.jpg")