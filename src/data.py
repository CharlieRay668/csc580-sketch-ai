"""Data handling utilities for Quick, Draw! dataset experiments.

The Google Quick, Draw! dataset contains 50M drawings across 345 categories.
This module provides utilities to download and load the dataset for classification tasks.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class QuickDrawDataset(Dataset):
    """Google Quick, Draw! dataset.
    
    The Quick, Draw! dataset contains 28x28 grayscale images of hand-drawn sketches
    across 345 categories. Data is stored in .npy format.
    
    Dataset source: https://github.com/googlecreativelab/quickdraw-dataset
    
    Args:
        root: Root directory for data storage
        categories: List of category names to load (default: None loads all 345)
        max_per_class: Maximum samples per class (default: 10000)
        transform: Torchvision transforms to apply
        download: Whether to download if not present
    """
    
    def __init__(
        self,
        root: str,
        categories: Optional[list] = None,
        max_per_class: int = 10000,
        transform=None,
        download: bool = True
    ):
        self.root = Path(root) / "quickdraw"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_per_class = max_per_class
        self.transform = transform
        
        # Default categories (subset for testing, expand for full dataset)
        if categories is None:
            self.categories = [
                'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
                'animal migration', 'ant', 'anvil', 'apple', 'arm',
                'asparagus', 'axe', 'backpack', 'banana', 'bandage',
                'barn', 'baseball', 'baseball bat', 'basket', 'basketball'
            ]
        else:
            self.categories = categories
        
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        if download:
            self._download_data()
        
        self._load_data()
    
    def _download_data(self):
        """Download Quick, Draw! .npy files for specified categories."""
        base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
        
        for category in self.categories:
            filename = f"{category.replace(' ', '%20')}.npy"
            filepath = self.root / f"{category}.npy"
            
            if filepath.exists():
                print(f"✓ {category} already downloaded")
                continue
            
            url = base_url + filename
            print(f"Downloading {category}...")
            
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ Downloaded {category}")
            except Exception as e:
                print(f"✗ Failed to download {category}: {e}")
    
    def _load_data(self):
        """Load .npy files into memory."""
        self.data = []
        self.labels = []
        
        for idx, category in enumerate(self.categories):
            filepath = self.root / f"{category}.npy"
            
            if not filepath.exists():
                print(f"Warning: {category}.npy not found, skipping...")
                continue
            
            # Load numpy array (shape: [N, 784] for 28x28 flattened images)
            try:
                cat_data = np.load(filepath)
                # Limit samples per class
                cat_data = cat_data[:self.max_per_class]
                
                self.data.append(cat_data)
                self.labels.extend([idx] * len(cat_data))
                
                print(f"Loaded {len(cat_data)} samples from {category}")
            except Exception as e:
                print(f"Error loading {category}: {e}")
        
        # Concatenate all categories
        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
        else:
            raise ValueError("No data loaded! Check if files exist and download=True")
        
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"\nTotal dataset size: {len(self.data)} images across {len(self.categories)} categories")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get flattened image and reshape to 28x28
        img_flat = self.data[idx]
        img = img_flat.reshape(28, 28).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img, mode='L')  # 'L' for grayscale
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label


class DataConfig:
    """Configuration for data loading."""
    
    def __init__(
        self,
        dataset="quickdraw",
        batch_size=128,
        data_root="./data",
        num_workers=4,
        categories=None,
        max_per_class=10000,
        train_split=0.8,
        image_size=28,
        augment=False
    ):
        """
        Args:
            dataset: Dataset name (only 'quickdraw' supported)
            batch_size: Batch size for data loaders
            data_root: Root directory for data storage
            num_workers: Number of worker processes for data loading
            categories: List of Quick, Draw! categories (None for default subset)
            max_per_class: Maximum samples per class
            train_split: Fraction of data for training (rest for validation)
            image_size: Target image size (default: 28)
            augment: Whether to apply data augmentation
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_root = Path(data_root)
        self.num_workers = num_workers
        self.categories = categories
        self.max_per_class = max_per_class
        self.train_split = train_split
        self.image_size = image_size
        self.augment = augment
        
        # Quick, Draw! is grayscale
        self.channels = 1
    
    def get_transforms(self, train=True):
        """Get data transforms for Quick, Draw! dataset."""
        transform_list = []
        
        # Resize if needed
        if self.image_size != 28:
            transform_list.append(transforms.Resize(self.image_size))
        
        # Data augmentation for training
        if train and self.augment:
            transform_list.extend([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        if self.dataset.lower() != "quickdraw":
            raise ValueError(f"Only 'quickdraw' dataset supported, got: {self.dataset}")
        
        # Load full dataset
        full_dataset = QuickDrawDataset(
            root=str(self.data_root),
            categories=self.categories,
            max_per_class=self.max_per_class,
            transform=self.get_transforms(train=True),
            download=True
        )
        
        # Split into train/val
        train_size = int(self.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply different transforms for validation (no augmentation)
        val_transform = self.get_transforms(train=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"\nDataloaders created:")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        
        return train_loader, val_loader


# Full list of 345 Quick, Draw! categories for reference
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
    'cactus', 'cake', 'calculator', 'calendar', 'camel',
    'camera', 'camouflage', 'campfire', 'candle', 'cannon',
    'canoe', 'car', 'carrot', 'castle', 'cat',
    'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier',
    'church', 'circle', 'clarinet', 'clock', 'cloud',
    'coffee cup', 'compass', 'computer', 'cookie', 'cooler',
    'couch', 'cow', 'crab', 'crayon', 'crocodile',
    'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher',
    'diving board', 'dog', 'dolphin', 'donut', 'door',
    'dragon', 'dresser', 'drill', 'drums', 'duck',
    'dumbbell', 'ear', 'elbow', 'elephant', 'envelope',
    'eraser', 'eye', 'eyeglasses', 'face', 'fan',
    'feather', 'fence', 'finger', 'fire hydrant', 'fireplace',
    'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops',
    'floor lamp', 'flower', 'flying saucer', 'foot', 'fork',
    'frog', 'frying pan', 'garden', 'garden hose', 'giraffe',
    'goatee', 'golf club', 'grapes', 'grass', 'guitar',
    'hamburger', 'hammer', 'hand', 'harp', 'hat',
    'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon',
    'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon',
    'hot dog', 'hot tub', 'hourglass', 'house', 'house plant',
    'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo',
    'key', 'keyboard', 'knee', 'knife', 'ladder',
    'lantern', 'laptop', 'leaf', 'leg', 'light bulb',
    'lighter', 'lighthouse', 'lightning', 'line', 'lion',
    'lipstick', 'lobster', 'lollipop', 'mailbox', 'map',
    'marker', 'matches', 'megaphone', 'mermaid', 'microphone',
    'microwave', 'monkey', 'moon', 'mosquito', 'motorbike',
    'mountain', 'mouse', 'moustache', 'mouth', 'mug',
    'mushroom', 'nail', 'necklace', 'nose', 'ocean',
    'octagon', 'octopus', 'onion', 'oven', 'owl',
    'paint can', 'paintbrush', 'palm tree', 'panda', 'pants',
    'paper clip', 'parachute', 'parrot', 'passport', 'peanut',
    'pear', 'peas', 'pencil', 'penguin', 'piano',
    'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple',
    'pizza', 'pliers', 'police car', 'pond', 'pool',
    'popsicle', 'postcard', 'potato', 'power outlet', 'purse',
    'rabbit', 'raccoon', 'radio', 'rain', 'rainbow',
    'rake', 'remote control', 'rhinoceros', 'rifle', 'river',
    'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
    'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver',
    'sea turtle', 'see saw', 'shark', 'sheep', 'shoe',
    'shorts', 'shovel', 'sink', 'skateboard', 'skull',
    'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake',
    'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock',
    'speedboat', 'spider', 'spoon', 'spreadsheet', 'square',
    'squiggle', 'squirrel', 'stairs', 'star', 'steak',
    'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove',
    'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase',
    'sun', 'swan', 'sweater', 'swing set', 'sword',
    'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear',
    'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower',
    'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe',
    'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
    'tractor', 'traffic light', 'train', 'tree', 'triangle',
    'trombone', 'truck', 'trumpet', 'umbrella', 'underwear',
    'van', 'vase', 'violin', 'washing machine', 'watermelon',
    'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle',
    'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'
]
