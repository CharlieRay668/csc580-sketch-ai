"""
Utility functions for loading models and working with guessing policies.
Provides helper functions to make model loading and policy creation easier.
"""

import torch
from pathlib import Path
from typing import Tuple, Optional
from src.models import create_model
from guesser.policies import GuessingPolicy, PolicyWrapper, create_policy


def detect_num_classes(checkpoint_path: str, device: torch.device = torch.device('cpu')) -> int:
    """Detect the number of classes from a model checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Number of classes the model was trained on
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check different possible locations based on model architecture
    state_dict = checkpoint['model_state_dict']
    
    if 'head.weight' in state_dict:
        # MLP architecture
        return state_dict['head.weight'].shape[0]
    elif 'fc.weight' in state_dict:
        # ResNet architecture
        return state_dict['fc.weight'].shape[0]
    elif 'mlp_head.1.weight' in state_dict:
        # ViT architecture (Sequential: LayerNorm + Linear)
        return state_dict['mlp_head.1.weight'].shape[0]
    else:
        # Fallback: search for the first 2D weight tensor (likely final layer)
        for key in state_dict.keys():
            if 'weight' in key and len(state_dict[key].shape) == 2:
                return state_dict[key].shape[0]
        
        raise ValueError(f"Could not detect num_classes from checkpoint at {checkpoint_path}")


def load_model_with_policy(
    model_path: str,
    model_type: str,
    policy_type: str,
    device: Optional[torch.device] = None,
    policy_kwargs: Optional[dict] = None
) -> Tuple[PolicyWrapper, int]:
    """Load a model and wrap it with a guessing policy.
    
    This is a convenience function that:
    1. Automatically detects the number of classes from the checkpoint
    2. Loads the model
    3. Creates the specified policy
    4. Returns a ready-to-use PolicyWrapper
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        model_type: Model architecture ('mlp', 'resnet18', or 'vit')
        policy_type: Policy type ('confidence', 'time', or 'learned')
        device: Device to load model on (default: auto-detect)
        policy_kwargs: Additional arguments for the policy (e.g., {'threshold': 0.85})
        
    Returns:
        Tuple of (PolicyWrapper, num_classes)
        
    Example:
        >>> wrapper, num_classes = load_model_with_policy(
        ...     'models/resnet18_best.pth',
        ...     'resnet18',
        ...     'confidence',
        ...     policy_kwargs={'threshold': 0.9}
        ... )
        >>> should_guess, pred, conf, logits = wrapper.predict(image, elapsed_time=3.0)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if policy_kwargs is None:
        policy_kwargs = {}
    
    # Detect number of classes
    num_classes = detect_num_classes(model_path, device)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create policy (add num_classes for learned policy)
    if policy_type == 'learned':
        policy_kwargs['num_classes'] = num_classes
    
    policy = create_policy(policy_type, **policy_kwargs)
    
    # Create wrapper
    wrapper = PolicyWrapper(model, policy, device)
    
    return wrapper, num_classes


def load_all_models_with_policy(
    models_dir: str = "models",
    policy_type: str = "confidence",
    device: Optional[torch.device] = None,
    policy_kwargs: Optional[dict] = None
) -> dict:
    """Load all three models (MLP, ResNet-18, ViT) with the same policy.
    
    Args:
        models_dir: Directory containing model checkpoints
        policy_type: Policy type to use for all models
        device: Device to load models on (default: auto-detect)
        policy_kwargs: Policy configuration
        
    Returns:
        Dictionary mapping model names to (PolicyWrapper, num_classes) tuples
        
    Example:
        >>> wrappers = load_all_models_with_policy(
        ...     policy_type='confidence',
        ...     policy_kwargs={'threshold': 0.85}
        ... )
        >>> for name, (wrapper, num_classes) in wrappers.items():
        ...     print(f"{name}: {num_classes} classes")
        ...     should_guess, pred, conf, _ = wrapper.predict(image, elapsed_time=2.0)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if policy_kwargs is None:
        policy_kwargs = {}
    
    models_dir = Path(models_dir)
    model_configs = [
        ("MLP", "mlp", models_dir / "mlp_best.pth"),
        ("ResNet-18", "resnet18", models_dir / "resnet18_best.pth"),
        ("ViT", "vit", models_dir / "vit_best.pth"),
    ]
    
    wrappers = {}
    
    for name, model_type, model_path in model_configs:
        if not model_path.exists():
            print(f"Warning: {name} model not found at {model_path}")
            continue
        
        try:
            wrapper, num_classes = load_model_with_policy(
                str(model_path),
                model_type,
                policy_type,
                device,
                policy_kwargs
            )
            wrappers[name] = (wrapper, num_classes)
            print(f"Loaded {name} with {num_classes} classes")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return wrappers


def compare_policies_on_model(
    model_path: str,
    model_type: str,
    policies_config: dict,
    device: Optional[torch.device] = None
) -> dict:
    """Load one model with multiple different policies for comparison.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model architecture
        policies_config: Dict mapping policy names to (policy_type, policy_kwargs) tuples
        device: Device to load on
        
    Returns:
        Dictionary mapping policy names to PolicyWrapper instances
        
    Example:
        >>> policies = {
        ...     'Cautious': ('confidence', {'threshold': 0.95}),
        ...     'Moderate': ('confidence', {'threshold': 0.85}),
        ...     'Quick': ('time', {'wait_time': 3.0}),
        ...     'Learned': ('learned', {}),
        ... }
        >>> wrappers = compare_policies_on_model(
        ...     'models/resnet18_best.pth',
        ...     'resnet18',
        ...     policies
        ... )
        >>> for name, wrapper in wrappers.items():
        ...     result = test_with_policy(wrapper)
        ...     print(f"{name}: {result}")
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Detect number of classes
    num_classes = detect_num_classes(model_path, device)
    
    # Load model once
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create wrappers with different policies
    wrappers = {}
    
    for name, (policy_type, policy_kwargs) in policies_config.items():
        # Add num_classes for learned policy
        kwargs = policy_kwargs.copy()
        if policy_type == 'learned':
            kwargs['num_classes'] = num_classes
        
        policy = create_policy(policy_type, **kwargs)
        wrapper = PolicyWrapper(model, policy, device)
        wrappers[name] = wrapper
    
    return wrappers


# Preset policy configurations for common use cases
POLICY_PRESETS = {
    'cautious': ('confidence', {'threshold': 0.95}),
    'moderate': ('confidence', {'threshold': 0.85}),
    'aggressive': ('confidence', {'threshold': 0.70}),
    'quick': ('time', {'wait_time': 3.0, 'use_best_guess': True}),
    'patient': ('time', {'wait_time': 8.0, 'use_best_guess': True}),
    'instant': ('time', {'wait_time': 0.5, 'use_best_guess': False}),
    'learned': ('learned', {'threshold': 0.5, 'include_confidence': True, 'include_time': True}),
}


def load_model_with_preset(
    model_path: str,
    model_type: str,
    preset_name: str,
    device: Optional[torch.device] = None
) -> Tuple[PolicyWrapper, int]:
    """Load a model with a preset policy configuration.
    
    Available presets:
        - 'cautious': High confidence threshold (95%)
        - 'moderate': Medium confidence threshold (85%)
        - 'aggressive': Low confidence threshold (70%)
        - 'quick': Fast time-based (3 seconds)
        - 'patient': Slow time-based (8 seconds)
        - 'instant': Very fast time-based (0.5 seconds)
        - 'learned': Neural network policy
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model architecture
        preset_name: Name of preset configuration
        device: Device to load on
        
    Returns:
        Tuple of (PolicyWrapper, num_classes)
        
    Example:
        >>> wrapper, num_classes = load_model_with_preset(
        ...     'models/resnet18_best.pth',
        ...     'resnet18',
        ...     'moderate'
        ... )
    """
    if preset_name not in POLICY_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(POLICY_PRESETS.keys())}"
        )
    
    policy_type, policy_kwargs = POLICY_PRESETS[preset_name]
    
    return load_model_with_policy(
        model_path,
        model_type,
        policy_type,
        device,
        policy_kwargs
    )
