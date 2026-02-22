"""Parallel preprocessing of stroke detection for Quick Draw dataset.

This script processes the entire dataset using multiprocessing to extract
strokes from each image. Results are saved to HDF5 for fast training.
"""

import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
import sys
import urllib.request

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.stroke_detection import StrokeDetector

# All 345 Quick Draw categories
ALL_QUICKDRAW_CATEGORIES = [
    'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
    'animal migration', 'ant', 'anvil', 'apple', 'arm',
    'asparagus', 'axe', 'backpack', 'banana', 'bandage',
    'barn', 'baseball', 'baseball bat', 'basket', 'basketball',
    'bat', 'bathtub', 'beach', 'bear', 'beard',
    'bed', 'bee', 'belt', 'bench', 'bicycle',
    'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry',
    'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet',
    'brain', 'bread', 'bridge', 'broccoli', 'broom',
    'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
]  # Truncated for brevity - using first 50 categories


def download_category(category, data_dir):
    """Download .npy file for a category if not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / f"{category}.npy"
    if filepath.exists():
        print(f"✓ {category} already downloaded")
        return filepath
        
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category.replace(' ', '%20')}.npy"
    print(f"Downloading {category}...")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ {category} downloaded")
        return filepath
    except Exception as e:
        print(f"✗ Failed to download {category}: {e}")
        return None


def load_quickdraw_data(categories, data_dir, max_per_class=10000):
    """Load Quick Draw data without PyTorch dependencies.
    
    Returns:
        images: numpy array of shape (N, 28, 28)
        labels: numpy array of shape (N,)
    """
    all_images = []
    all_labels = []
    
    for label_idx, category in enumerate(categories):
        filepath = download_category(category, data_dir)
        if filepath is None:
            continue
            
        # Load numpy data
        data = np.load(filepath)
        
        # Take subset
        n_samples = min(len(data), max_per_class)
        data = data[:n_samples]
        
        # Reshape from (N, 784) to (N, 28, 28)
        images = data.reshape(-1, 28, 28).astype(np.uint8)
        labels = np.full(n_samples, label_idx, dtype=np.int64)
        
        all_images.append(images)
        all_labels.append(labels)
        
        print(f"Loaded {n_samples} samples from {category}")
    
    return np.concatenate(all_images), np.concatenate(all_labels)


def process_single_image(args):
    """Process one image - designed for multiprocessing.
    
    Args:
        args: Tuple of (image, label, index, detector_config)
    
    Returns:
        Tuple of (index, strokes_data, label)
    """
    image, label, idx, detector_config = args
    
    # Create detector (each process needs its own)
    detector = StrokeDetector(**detector_config)
    
    try:
        # Detect strokes
        strokes = detector.detect_strokes(image)
        
        # Order strokes
        ordered_strokes = detector.order_strokes_connected(strokes)
        
        # Pack strokes into serializable format
        # Store as list of (stroke_mask, num_pixels) for compression
        strokes_data = []
        for stroke in ordered_strokes:
            # Get coordinates of stroke pixels
            coords = np.argwhere(stroke > 0)
            values = stroke[coords[:, 0], coords[:, 1]]
            strokes_data.append({
                'coords': coords.astype(np.int16),
                'values': values.astype(np.uint8),
                'shape': stroke.shape
            })
        
        return (idx, strokes_data, label)
    
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        return (idx, [], label)


def preprocess_dataset(
    num_classes: int = 50,
    samples_per_class: int = 5000,
    num_workers: int = 64,
    output_file: str = 'data/preprocessed_strokes.h5',
    detector_config: dict = None
):
    """Preprocess entire dataset with parallel stroke detection.
    
    Args:
        num_classes: Number of classes to process
        samples_per_class: Max samples per class
        num_workers: Number of parallel workers
        output_file: Path to save HDF5 file
        detector_config: StrokeDetector configuration
    """
    
    if detector_config is None:
        detector_config = {
            'min_stroke_pixels': 20,
            'connectivity': 8,
            'erosion_iterations': 2,
            'separation_strength': 3,
            'use_skeleton_split': True
        }
    
    print("="*60)
    print("STROKE PREPROCESSING - PARALLEL")
    print("="*60)
    print(f"Classes: {num_classes}")
    print(f"Samples/class: {samples_per_class}")
    print(f"Total images: {num_classes * samples_per_class}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_file}")
    print("="*60)
    
    # Get categories
    categories = ALL_QUICKDRAW_CATEGORIES[:num_classes]
    
    # Load dataset
    print("\nLoading dataset...")
    images, labels = load_quickdraw_data(
        categories=categories,
        data_dir='data/quickdraw',
        max_per_class=samples_per_class
    )
    
    print(f"\nTotal dataset size: {len(images)} images across {num_classes} categories")
    print(f"Dataset loaded: {len(images)} images")
    
    # Prepare arguments for parallel processing
    print("\nPreparing data for parallel processing...")
    process_args = []
    for idx in range(len(images)):
        image = images[idx]
        label = labels[idx]
        process_args.append((image, label, idx, detector_config))
    
    # Process in parallel
    print(f"\nProcessing with {num_workers} workers...")
    print("This may take 30-60 minutes depending on CPU speed...")
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, process_args),
            total=len(process_args),
            desc="Processing images",
            unit="img"
        ))
    
    # Save to HDF5
    print("\nSaving to HDF5...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # Store metadata
        f.attrs['num_images'] = len(results)
        f.attrs['num_classes'] = num_classes
        f.attrs['samples_per_class'] = samples_per_class
        f.attrs['categories'] = [cat.encode('utf-8') for cat in categories]
        
        # Create groups
        strokes_group = f.create_group('strokes')
        labels_dset = f.create_dataset('labels', (len(results),), dtype='i')
        num_strokes_dset = f.create_dataset('num_strokes', (len(results),), dtype='i')
        
        # Sort results by index
        results.sort(key=lambda x: x[0])
        
        # Store each image's strokes
        for idx, strokes_data, label in tqdm(results, desc="Saving to disk"):
            labels_dset[idx] = label
            num_strokes_dset[idx] = len(strokes_data)
            
            if len(strokes_data) > 0:
                # Create subgroup for this image
                img_group = strokes_group.create_group(f'img_{idx:06d}')
                
                for stroke_idx, stroke_info in enumerate(strokes_data):
                    stroke_subgroup = img_group.create_group(f'stroke_{stroke_idx}')
                    stroke_subgroup.create_dataset('coords', data=stroke_info['coords'], 
                                                  compression='gzip', compression_opts=4)
                    stroke_subgroup.create_dataset('values', data=stroke_info['values'],
                                                  compression='gzip', compression_opts=4)
                    stroke_subgroup.attrs['shape'] = stroke_info['shape']
    
    # Print statistics
    file_size = output_path.stat().st_size / (1024**3)
    num_strokes_total = sum(r[1] and len(r[1]) or 0 for r in results)
    avg_strokes = num_strokes_total / len(results)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Output file: {output_file}")
    print(f"File size: {file_size:.2f} GB")
    print(f"Total images: {len(results)}")
    print(f"Total strokes: {num_strokes_total}")
    print(f"Avg strokes/image: {avg_strokes:.1f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Quick Draw dataset for stroke-based training')
    parser.add_argument('--num-classes', type=int, default=50,
                       help='Number of classes to process (default: 50)')
    parser.add_argument('--samples-per-class', type=int, default=5000,
                       help='Max samples per class (default: 5000)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPUs)')
    parser.add_argument('--output', type=str, default='data/preprocessed_strokes.h5',
                       help='Output HDF5 file path')
    
    args = parser.parse_args()
    
    # Use all CPUs if not specified
    num_workers = args.num_workers or cpu_count()
    
    preprocess_dataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        num_workers=num_workers,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
