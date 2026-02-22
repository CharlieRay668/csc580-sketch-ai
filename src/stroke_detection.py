"""
Stroke detection and progressive reveal for sketches.
Detects individual strokes in a drawing and reveals them sequentially.
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage.morphology import skeletonize


class StrokeDetector:
    """Detects and extracts individual strokes from a sketch."""
    
    def __init__(
        self,
        min_stroke_pixels: int = 10,
        connectivity: int = 8,
        erosion_iterations: int = 2,
        separation_strength: int = 3,
        use_skeleton_split: bool = True
    ):
        self.min_stroke_pixels = min_stroke_pixels
        self.connectivity = connectivity
        self.erosion_iterations = erosion_iterations
        self.separation_strength = separation_strength
        self.use_skeleton_split = use_skeleton_split
    
    def detect_strokes(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect individual strokes in a sketch."""
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        binary = (image > 0).astype(np.uint8) * 255
        
        if self.erosion_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=self.erosion_iterations)
        
        if self.separation_strength > 0:
            dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, cv2.THRESH_BINARY)
            sure_fg = sure_fg.astype(np.uint8)
            
            kernel = np.ones((self.separation_strength, self.separation_strength), np.uint8)
            sure_bg = cv2.dilate(eroded, kernel, iterations=1)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            image_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(image_3ch, markers)
            
            strokes = []
            unique_labels = np.unique(markers)
            for label in unique_labels:
                if label <= 1:
                    continue
                
                seed_mask = (markers == label).astype(np.uint8) * 255
                stroke_mask = self._expand_to_original(seed_mask, binary)
                
                if np.sum(stroke_mask > 0) >= self.min_stroke_pixels:
                    strokes.append(stroke_mask)
        
        all_strokes_combined = np.zeros_like(binary)
        for stroke in strokes:
            all_strokes_combined = cv2.bitwise_or(all_strokes_combined, stroke)
        
        missing_pixels = cv2.bitwise_and(binary, cv2.bitwise_not(all_strokes_combined))
        missing_count = np.sum(missing_pixels > 0)
        
        if missing_count > 0:
            print(f"  Found {missing_count} missing pixels, assigning to nearest strokes...")
            
            if len(strokes) == 0:
                strokes.append(missing_pixels)
            else:
                missing_y, missing_x = np.where(missing_pixels > 0)
                
                for my, mx in zip(missing_y, missing_x):
                    min_dist = float('inf')
                    nearest_idx = 0
                    
                    for idx, stroke in enumerate(strokes):
                        stroke_y, stroke_x = np.where(stroke > 0)
                        if len(stroke_y) > 0:
                            distances = np.sqrt((stroke_y - my)**2 + (stroke_x - mx)**2)
                            min_stroke_dist = np.min(distances)
                            if min_stroke_dist < min_dist:
                                min_dist = min_stroke_dist
                                nearest_idx = idx
                    
                    strokes[nearest_idx][my, mx] = 255
        
        final_check = np.zeros_like(binary)
        for stroke in strokes:
            final_check = cv2.bitwise_or(final_check, stroke)
        still_missing = cv2.bitwise_and(binary, cv2.bitwise_not(final_check))
        if np.sum(still_missing > 0) > 0:
            print(f"  WARNING: {np.sum(still_missing > 0)} pixels still not covered after assignment!")
        
        if len(strokes) == 1 and self.use_skeleton_split:
            split_strokes = self._split_stroke_by_skeleton(strokes[0])
            if len(split_strokes) > 1:
                strokes = split_strokes
        
        # Convert binary stroke masks to grayscale strokes with original pixel values
        grayscale_strokes = []
        for stroke_mask in strokes:
            grayscale_stroke = np.zeros_like(image, dtype=np.uint8)
            grayscale_stroke[stroke_mask > 0] = image[stroke_mask > 0]
            grayscale_strokes.append(grayscale_stroke)
        
        return grayscale_strokes
    
    def _expand_to_original(self, seed_mask: np.ndarray, original_binary: np.ndarray) -> np.ndarray:
        """Expand seed mask to capture all connected pixels in original binary."""
        expanded = seed_mask.copy()
        kernel = np.ones((3, 3), np.uint8)
        
        for _ in range(20):
            dilated = cv2.dilate(expanded, kernel, iterations=1)
            new_expanded = cv2.bitwise_and(dilated, original_binary)
            if np.array_equal(new_expanded, expanded):
                break
            expanded = new_expanded
        
        return expanded
    
    def _split_stroke_by_skeleton(self, stroke_mask: np.ndarray) -> List[np.ndarray]:
        """Split a single stroke into multiple parts using skeleton analysis."""
        # Skeletonize the stroke
        skeleton = skeletonize(stroke_mask > 0)
        
        # Find branch points and endpoints
        # Branch point: has more than 2 neighbors
        # Endpoint: has exactly 1 neighbor
        def count_neighbors(y, x, skel):
            h, w = skel.shape
            count = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                        count += 1
            return count
        
        # Find all skeleton points
        y_coords, x_coords = np.where(skeleton)
        
        # Identify branch points (>2 neighbors)
        branch_points = []
        for y, x in zip(y_coords, x_coords):
            if count_neighbors(y, x, skeleton) > 2:
                branch_points.append((y, x))
        
        # If no branch points, try splitting by distance from center
        if len(branch_points) == 0:
            return self._split_by_distance(stroke_mask, num_parts=3)
        
        # Remove branch points from skeleton to split it
        split_skeleton = skeleton.copy()
        for y, x in branch_points:
            # Remove a small region around branch point
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < split_skeleton.shape[0] and 0 <= nx < split_skeleton.shape[1]:
                        split_skeleton[ny, nx] = False
        
        # Find connected components in the split skeleton
        labeled_skeleton, num_segments = ndimage.label(split_skeleton)
        
        # Dilate each segment back to stroke width
        strokes = []
        for segment_id in range(1, num_segments + 1):
            segment_mask = (labeled_skeleton == segment_id).astype(np.uint8) * 255
            
            # Dilate to recover stroke width
            dilated = cv2.dilate(segment_mask, np.ones((5, 5), np.uint8), iterations=2)
            
            # Intersect with original stroke to get clean edges
            final_stroke = cv2.bitwise_and(dilated, stroke_mask)
            
            if np.sum(final_stroke > 0) >= self.min_stroke_pixels:
                strokes.append(final_stroke)
        
        # If splitting produced good results, return them; otherwise return original
        return strokes if len(strokes) > 1 else [stroke_mask]
    
    def _split_by_distance(self, stroke_mask: np.ndarray, num_parts: int = 3) -> List[np.ndarray]:
        """Split a stroke into parts based on distance along the shape."""
        # Find the skeleton path
        skeleton = skeletonize(stroke_mask > 0)
        y_coords, x_coords = np.where(skeleton)
        
        if len(y_coords) < num_parts * 5:  # Too short to split
            return [stroke_mask]
        
        # Find endpoints (1 neighbor)
        def count_neighbors(y, x, skel):
            h, w = skel.shape
            count = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                        count += 1
            return count
        
        endpoints = []
        for y, x in zip(y_coords, x_coords):
            if count_neighbors(y, x, skeleton) == 1:
                endpoints.append((y, x))
        
        # If we have endpoints, split along the skeleton
        if len(endpoints) >= 2:
            # Create distance map from first endpoint
            start = endpoints[0]
            dist_map = np.zeros_like(skeleton, dtype=float)
            dist_map[start] = 1
            
            
            # Simple approach: divide skeleton into regions
            skeleton_points = list(zip(y_coords, x_coords))
            total_points = len(skeleton_points)
            points_per_part = total_points // num_parts
            
            # Create masks for each part
            strokes = []
            for i in range(num_parts):
                part_skeleton = np.zeros_like(skeleton, dtype=np.uint8)
                start_idx = i * points_per_part
                end_idx = (i + 1) * points_per_part if i < num_parts - 1 else total_points
                
                for j in range(start_idx, end_idx):
                    y, x = skeleton_points[j]
                    part_skeleton[y, x] = 255
                
                # Dilate back to stroke width
                dilated = cv2.dilate(part_skeleton, np.ones((5, 5), np.uint8), iterations=2)
                final_stroke = cv2.bitwise_and(dilated, stroke_mask)
                
                if np.sum(final_stroke > 0) >= self.min_stroke_pixels:
                    strokes.append(final_stroke)
            
            return strokes if len(strokes) > 1 else [stroke_mask]
        
        return [stroke_mask]
    
    def order_strokes_connected(self, strokes: List[np.ndarray]) -> List[np.ndarray]:
        """Order strokes by connectivity (like drawing)."""
        if len(strokes) <= 1:
            return strokes
        
        ordered = []
        remaining = set(range(len(strokes)))
        
        centroids = []
        for stroke in strokes:
            y_coords, x_coords = np.where(stroke > 0)
            if len(y_coords) > 0:
                centroids.append((np.mean(y_coords), np.mean(x_coords)))
            else:
                centroids.append((0, 0))
        
        distances_to_origin = [y + x for y, x in centroids]
        current_idx = int(np.argmin(distances_to_origin))
        
        while remaining:
            ordered.append(strokes[current_idx])
            remaining.discard(current_idx)
            
            if not remaining:
                break
            
            current_stroke = strokes[current_idx]
            dilated_current = cv2.dilate(current_stroke, np.ones((3, 3), np.uint8), iterations=1)
            
            best_overlap = 0
            best_idx = None
            
            for idx in remaining:
                overlap = cv2.bitwise_and(dilated_current, strokes[idx])
                overlap_count = np.sum(overlap > 0)
                
                if overlap_count > best_overlap:
                    best_overlap = overlap_count
                    best_idx = idx
            
            if best_idx is not None and best_overlap > 0:
                current_idx = best_idx
            else:
                current_y, current_x = centroids[current_idx]
                min_dist = float('inf')
                for idx in remaining:
                    y, x = centroids[idx]
                    dist = np.sqrt((y - current_y)**2 + (x - current_x)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                current_idx = best_idx
        
        return ordered


class ProgressiveStrokeRevealer:
    """Progressively reveals strokes from a sketch."""
    
    def __init__(self, detector: Optional[StrokeDetector] = None, 
                 erosion_iterations: int = 2, separation_strength: int = 3):
        if detector is None:
            detector = StrokeDetector(
                min_stroke_pixels=20, connectivity=8,
                erosion_iterations=erosion_iterations,
                separation_strength=separation_strength
            )
        self.detector = detector
        self.strokes = None
        self.current_index = 0
    
    def load_image(self, image: np.ndarray):
        """Load and process an image into strokes."""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                self.original_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:
                self.original_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            self.original_image = image.copy()
        
        if self.original_image.dtype == np.float32 or self.original_image.dtype == np.float64:
            self.original_image = (self.original_image * 255).astype(np.uint8)
        
        strokes = self.detector.detect_strokes(image)
        self.strokes = self.detector.order_strokes_connected(strokes)
        self.current_index = 0
        print(f"Detected {len(self.strokes)} strokes")
    
    def get_progressive_frames(self, background: str = 'black', preserve_grayscale: bool = True) -> List[np.ndarray]:
        """Generate frames with progressively more strokes."""
        if self.strokes is None or len(self.strokes) == 0:
            return []
        
        height, width = self.strokes[0].shape
        bg_value = 255 if background == 'white' else 0
        
        frames = []
        cumulative = np.full((height, width), bg_value, dtype=np.uint8)
        frames.append(cumulative.copy())
        
        for stroke in self.strokes:
            mask = stroke > 0
            
            if preserve_grayscale and hasattr(self, 'original_image'):
                cumulative[mask] = self.original_image[mask]
            else:
                fg_value = 0 if background == 'white' else 255
                cumulative[mask] = fg_value
            
            frames.append(cumulative.copy())
        
        return frames
    
    def reset(self):
        """Reset to first frame."""
        self.current_index = 0
    
    def get_num_strokes(self) -> int:
        """Get total number of strokes."""
        return len(self.strokes) if self.strokes else 0


def create_stroke_gif(
    image_path: str,
    output_path: str = "stroke_reveal.gif",
    duration_per_frame: int = 200,
    background: str = 'white',
    resize: Optional[Tuple[int, int]] = None,
    erosion_iterations: int = 2,
    separation_strength: int = 3
):
    """Create a GIF showing progressive stroke reveal."""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    revealer = ProgressiveStrokeRevealer(
        erosion_iterations=erosion_iterations,
        separation_strength=separation_strength
    )
    revealer.load_image(img_array)
    
    frames = revealer.get_progressive_frames(background=background, preserve_grayscale=True)
    
    if len(frames) == 0:
        print("No strokes detected!")
        return
    
    if resize is not None:
        resized_frames = []
        for frame in frames:
            frame_img = Image.fromarray(frame)
            
            resized_img = frame_img.resize(resize, Image.Resampling.NEAREST)
            resized_frames.append(np.array(resized_img))
        frames = resized_frames
    
    print(f"Generated {len(frames)} frames")
    
    # Convert to PIL images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_per_frame,
        loop=0
    )
    
    print(f"Saved GIF to {output_path}")
    print(f"Total strokes: {len(frames) - 1}")  # -1 for initial blank frame


def download_test_image(save_path: str = "test_sketch.png"):
    """Download a test sketch image from Quick Draw dataset.
    
    Args:
        save_path: Where to save the image
    """
    from src.data import ALL_QUICKDRAW_CATEGORIES
    import random
    
    category = random.choice(ALL_QUICKDRAW_CATEGORIES[:50])
    data_path = f"data/quickdraw/{category}.npy"
    
    try:
        data = np.load(data_path)

        idx = random.randint(0, min(100, len(data) - 1))
        sketch = data[idx].reshape(28, 28)
        
        img = Image.fromarray(sketch.astype(np.uint8))
        img.save(save_path)
        
        return save_path, category
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None, None
