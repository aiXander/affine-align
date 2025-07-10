import cv2
import numpy as np
# from scipy.optimize import differential_evolution  # Replaced with CMA-ES
import cma
import os
from typing import Tuple, Optional
import warnings
import time
warnings.filterwarnings('ignore')

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

class VisualAligner:
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.best_score = float('inf')
        self.best_params = None
        
    def normalize_params(self, params: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Convert parameters from original space to normalized space for CMA-ES."""
        tx, ty, scale_x, scale_y, rotation = params
        h, w = img_shape
        
        # Normalize translation by image dimensions (makes it relative to image size)
        norm_tx = tx / w
        norm_ty = ty / h
        
        # Normalize scale by centering around 0 (1.0 becomes 0.0)
        norm_scale_x = scale_x - 1.0
        norm_scale_y = scale_y - 1.0
        
        # Normalize rotation by 180 degrees (makes ±180° become ±1.0)
        norm_rotation = rotation / 180.0
        
        return np.array([norm_tx, norm_ty, norm_scale_x, norm_scale_y, norm_rotation])
    
    def denormalize_params(self, norm_params: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Convert parameters from normalized space back to original space."""
        norm_tx, norm_ty, norm_scale_x, norm_scale_y, norm_rotation = norm_params
        h, w = img_shape
        
        # Denormalize translation
        tx = norm_tx * w
        ty = norm_ty * h
        
        # Denormalize scale
        scale_x = norm_scale_x + 1.0
        scale_y = norm_scale_y + 1.0
        
        # Denormalize rotation
        rotation = norm_rotation * 180.0
        
        return np.array([tx, ty, scale_x, scale_y, rotation])
        
    def preprocess_image(self, img: np.ndarray, bypass_processor: bool = False, invert_pixels: bool = False) -> np.ndarray:
        """Preprocess image to extract edge features."""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        if bypass_processor:
            # Skip Canny edge detection, just use the grayscale image
            processed = gray
        else:
            processor = "canny"
            
            if processor == "canny":
                # Use Canny edge detection
                processed = cv2.Canny(gray, 50, 150)

                # Make the canny edges thicker:
                processed = cv2.dilate(processed, np.ones((2, 2), np.uint8), iterations=1)
            else:
                raise ValueError(f"Invalid processor: {processor}")
        
        if invert_pixels:
            processed = 255 - processed
            
        return processed
    
    def resize_maintain_aspect(self, img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
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
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]."""
        img_float = img.astype(np.float32)
        img_float = img_float - np.min(img_float)
        img_float = img_float / np.max(img_float)
        return img_float

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
        img_a_float = self.normalize_image(img_a)
        img_b_float = self.normalize_image(img_b)
        
        # Calculate Gaussian blur sigma based on image size
        # Use the smaller dimension to ensure blur is appropriate for both dimensions
        min_dim = min(img_a.shape[0], img_a.shape[1])
        sigma = blur_fraction * min_dim
        
        # Apply Gaussian blur to both images
        img_a_blurred = gaussian_filter(img_a_float, sigma=sigma)
        img_b_blurred = gaussian_filter(img_b_float, sigma=sigma)

        img_a_blurred = self.normalize_image(img_a_blurred)
        img_b_blurred = self.normalize_image(img_b_blurred)

        # Save blurred images (disabled cause slow)
        #cv2.imwrite("img_a_blurred.jpg", img_a_blurred*255)
        #cv2.imwrite("img_b_blurred.jpg", img_b_blurred*255)
        
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
    
    def compute_center_of_mass(self, img: np.ndarray, threshold: int) -> Tuple[float, float]:
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
    
    def optimize_affine_params(self, img_photo_padded: np.ndarray, img_drawing: np.ndarray, 
                              pad_offset_x: int, pad_offset_y: int, threshold: int) -> np.ndarray:
        """Optimize affine parameters by matching centers of mass."""
        print("    Optimizing affine parameters...")
        
        # Compute centers of mass
        photo_cx, photo_cy = self.compute_center_of_mass(img_photo_padded, threshold)
        drawing_cx, drawing_cy = self.compute_center_of_mass(img_drawing, threshold)
        
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
        
        # Optimize affine parameters using padded images
        affine_params_padded = self.optimize_affine_params(padded_photo, img_drawing, 
                                                          pad_offset_x, pad_offset_y, threshold)
        
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
        """Fine-tune alignment using CMA-ES optimization with normalized parameters."""
        start_time = time.time()
        
        h_d, w_d = img_drawing.shape[:2]
        img_shape = (h_d, w_d)
        
        # Normalize initial parameters for CMA-ES
        initial_norm_params = self.normalize_params(initial_params, img_shape)
        
        # Compute initial score to compare against
        initial_warped = self.apply_affine_transform(img_photo, initial_params, (h_d, w_d))
        initial_score = self.perceptual_loss(initial_warped, img_drawing)
        print(f"    Initial (coarse) score: {initial_score:.6f}")
        
        # Progress tracking
        iteration_count = 0
        best_score = initial_score
        best_params = initial_params.copy()
        best_iteration = 0 
        loss_history = []
        
        def objective(norm_params):
            nonlocal iteration_count, best_score, best_params, best_iteration, loss_history
            iteration_count += 1
            
            # Denormalize parameters back to original space
            params = self.denormalize_params(norm_params, img_shape)
            
            warped = self.apply_affine_transform(img_photo, params, (h_d, w_d))
            score = self.perceptual_loss(warped, img_drawing)
            loss_history.append(score)
            
            # Track best parameters (store in original space)
            if score < best_score:
                best_score = score
                best_iteration = iteration_count
                best_params = params.copy()
            elif iteration_count % 20 == 0:
                print(f"    Iteration {iteration_count}: score: {score:.6f} (best: {best_score:.6f} at iteration {best_iteration})")
            
            return score
        
        # Define step sizes for normalized parameters
        # Since parameters are now normalized, we can use similar scales for all
        norm_sigma = 0.25  # This corresponds to reasonable variations in normalized space
        initial_sigma = np.array([norm_sigma] * 5)  # Same sigma for all normalized parameters
        
        # Define bounds in normalized space
        # These are much more reasonable now since all parameters are on similar scales
        bound_range = 0.5  # Allow ±0.5 in normalized space
        
        lower_bounds = initial_norm_params - bound_range
        upper_bounds = initial_norm_params + bound_range
        
        print(f"    Using normalized parameters for CMA-ES:")
        print(f"    Initial normalized params: {initial_norm_params}")
        print(f"    Normalized sigma: {norm_sigma} (same for all parameters)")
        print(f"    Normalized bounds: ±{bound_range} around initial values")
        print(f"    This corresponds to:")
        print(f"      Translation: ±{bound_range * w_d:.1f}px, ±{bound_range * h_d:.1f}px")
        print(f"      Scale: ±{bound_range * 100:.1f}%")
        print(f"      Rotation: ±{bound_range * 180:.1f}°")
        
        # Use CMA-ES for optimization
        optimization_start = time.time()
        
        # CMA-ES options - much simpler now with normalized parameters
        options = {
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'maxiter': 30,            # Maximum iterations
            'popsize': 15,            # Population size
            'seed': 42,               # Random seed for reproducibility
            'verb_disp': 0,           # Reduce verbosity
            'verb_log': 0,
            'CMA_stds': initial_sigma.tolist(),  # Initial step sizes (normalized)
            'tolx': 1e-4,             # Tolerance for parameter changes
            'tolfun': 1e-5            # Tolerance for function value changes
        }
        
        # Run CMA-ES optimization with normalized parameters
        try:
            es = cma.CMAEvolutionStrategy(initial_norm_params, norm_sigma, options)
            es.optimize(objective)
            
            # Get results and denormalize
            result_norm_params = es.result.xbest
            result_params = self.denormalize_params(result_norm_params, img_shape)
            final_score = es.result.fbest
            n_iterations = es.result.evaluations
            
        except Exception as e:
            print(f"    CMA-ES optimization failed: {e}")
            print(f"    Falling back to initial parameters")
            result_params = initial_params
            final_score = initial_score
            n_iterations = 0
        
        optimization_time = time.time() - optimization_start
        
        total_time = time.time() - start_time
        print(f"  Fine alignment completed in {total_time:.3f}s (optimization: {optimization_time:.3f}s)")
        print(f"  Final CMA-ES score: {final_score:.6f}, function evaluations: {n_iterations}")
        print(f"  Best score found: {best_score:.6f}")

        # Plot loss history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('CMA-ES Optimization Progress (Normalized Parameters)')
        plt.grid(True)
        plt.savefig("loss_history_cmaes_normalized.png")
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
    
    def align_images(self, photo_path: str, drawing_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Main alignment function."""
        total_start = time.time()
        print("Starting image alignment process...")
        
        # Load images
        img_photo = cv2.imread(photo_path)
        img_drawing = cv2.imread(drawing_path)

        # Convert RGBA to RGB
        img_photo = cv2.cvtColor(img_photo, cv2.COLOR_BGRA2BGR)
        img_drawing = cv2.cvtColor(img_drawing, cv2.COLOR_BGRA2BGR)
        
        if img_photo is None or img_drawing is None:
            raise ValueError("Could not load images")
        
        # Store original photo for final transformation
        original_photo = img_photo.copy()
        
        # Resize for processing
        resize_start = time.time()
        img_photo_resized, photo_scale = self.resize_maintain_aspect(img_photo, self.max_size)
        img_drawing_resized, drawing_scale = self.resize_maintain_aspect(img_drawing, self.max_size)
        resize_time = time.time() - resize_start
        print(f"  Photo: {img_photo.shape} -> {img_photo_resized.shape}")
        print(f"  Drawing: {img_drawing.shape} -> {img_drawing_resized.shape}")
        
        # Preprocess to edge space
        preprocess_start = time.time()
        # For photos: apply Canny edge detection
        edges_photo = self.preprocess_image(img_photo_resized, bypass_processor=False, invert_pixels=False)
        # For drawings: bypass Canny and invert pixels to match edge-like appearance
        edges_drawing = self.preprocess_image(img_drawing_resized, bypass_processor=True, invert_pixels=True)
        preprocess_time = time.time() - preprocess_start
        
        # Save preprocessed images for debugging
        cv2.imwrite("debug_edges_photo_base.jpg", edges_photo)
        cv2.imwrite("debug_edges_drawing.jpg", edges_drawing)
        cv2.imwrite("debug_edges_photo_zcompare.jpg", edges_drawing)
        
        # Coarse alignment
        coarse_start = time.time()
        coarse_params, coarse_score = self.coarse_alignment(edges_photo, edges_drawing)
        coarse_time = time.time() - coarse_start
        print(f"Coarse alignment score: {coarse_score:.6f}")
        

        fine_start = time.time()
        final_params = self.fine_alignment(edges_photo, edges_drawing, coarse_params)
        fine_time = time.time() - fine_start
        print(f"Fine alignment took {fine_time:.3f}s")
        
        # Save post-alignment warped image for comparison
        h_d, w_d = edges_drawing.shape[:2]
        post_alignment_warped = self.apply_affine_transform(edges_photo, final_params, (h_d, w_d))
        
        cv2.imwrite("debug_edges_photo_warped_2.jpg", post_alignment_warped)
        print("Post-fine-alignment warped photo saved as debug_post_fine_warped.jpg")
        
        # Scale parameters back to original resolution
        scaling_start = time.time()
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
        scaling_time = time.time() - scaling_start
        print(f"Final transformation took {scaling_time:.3f}s")
        
        total_time = time.time() - total_start
        print(f"\nTotal alignment process took {total_time:.3f}s")
        print(f"Time breakdown:")
        print(f"  Resizing: {resize_time:.3f}s ({100*resize_time/total_time:.1f}%)")
        print(f"  Preprocessing: {preprocess_time:.3f}s ({100*preprocess_time/total_time:.1f}%)")
        print(f"  Coarse alignment: {coarse_time:.3f}s ({100*coarse_time/total_time:.1f}%)")
        print(f"  Fine alignment: {fine_time:.3f}s ({100*fine_time/total_time:.1f}%)")
        print(f"  Final transform: {scaling_time:.3f}s ({100*scaling_time/total_time:.1f}%)")
        
        return aligned_photo, final_params


def align_image_pair(photo_path: str, drawing_path: str, output_path: Optional[str] = None, skip_fine_alignment: bool = False) -> str:
    """
    Align a photo to match a line drawing.
    
    Args:
        photo_path: Path to the photo image
        drawing_path: Path to the line drawing
        output_path: Optional path to save the aligned photo
        skip_fine_alignment: If True, only use coarse alignment (faster, sometimes better)
        
    Returns:
        Path to the aligned photo
    """
    aligner = VisualAligner()
    aligned_photo, params = aligner.align_images(photo_path, drawing_path)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(photo_path))[0]
        suffix = "_coarse_aligned" if skip_fine_alignment else "_aligned"
        output_path = f"{base_name}{suffix}.jpg"
    
    cv2.imwrite(output_path, aligned_photo)
    
    return output_path

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
            # Align the image pair
            result_path = align_image_pair(file_a, file_b, output_path)
            print(f"  ✓ Successfully aligned and saved to: {result_path}")
            successful_alignments += 1
        except Exception as e:
            print(f"  ✗ Failed to align: {e}")
            failed_alignments += 1
    
    print(f"\n=== Alignment Summary ===")
    print(f"Successful alignments: {successful_alignments}")
    print(f"Failed alignments: {failed_alignments}")
    print(f"Total pairs processed: {successful_alignments + failed_alignments}")


def main():
    # Example usage
    #photo_path = "1_photo.jpeg"  # Replace with your photo path
    #drawing_path = "1_drawing.jpeg"  # Replace with your drawing path
    
    #full_aligned_path = align_image_pair(photo_path, drawing_path)

    input_folder = "/home/rednax/Documents/datasets/good_styles/stitchly_final/Kontext/combi"
    input_folder = "/home/rednax/Documents/datasets/good_styles/stitchly_final/Kontext/test"
    align_folders(input_folder, "{}_rembg.png", input_folder, "{}_D2.jpg", input_folder, "{}_D3.jpg")


if __name__ == "__main__":
    main()