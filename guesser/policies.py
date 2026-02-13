import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class GuessingPolicy(ABC):
    """Base class for guessing policies."""
    
    @abstractmethod
    def should_guess(self, logits: torch.Tensor, elapsed_time: Optional[float] = None, 
                     **kwargs) -> Tuple[bool, Optional[int], Optional[float]]:
        """Returns (should_guess, predicted_class, confidence)."""
        pass
    
    def reset(self):
        """Reset internal state."""
        pass
    
    def _get_prediction(self, logits: torch.Tensor) -> Tuple[int, float]:
        """Extract prediction and confidence from logits."""
        probs = F.softmax(logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        confidence, predicted_class = torch.max(probs, dim=-1)
        return predicted_class.item(), confidence.item()


class ConfidenceThresholdPolicy(GuessingPolicy):
    """Guess when model confidence exceeds threshold."""
    
    def __init__(self, threshold: float = 0.9):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self._has_guessed = False
    
    def should_guess(self, logits: torch.Tensor, elapsed_time: Optional[float] = None, 
                     **kwargs) -> Tuple[bool, Optional[int], Optional[float]]:
        if self._has_guessed:
            return False, None, None
        
        pred, conf = self._get_prediction(logits)
        if conf >= self.threshold:
            self._has_guessed = True
            return True, pred, conf
        return False, None, None
    
    def reset(self):
        self._has_guessed = False


class TimeBasedPolicy(GuessingPolicy):
    """Guess after a certain number of strokes."""
    
    def __init__(self, num_strokes: int = 5, use_best_guess: bool = False):
        if num_strokes <= 0:
            raise ValueError(f"num_strokes must be positive, got {num_strokes}")
        self.num_strokes = num_strokes
        self.use_best_guess = use_best_guess
        self._has_guessed = False
        self._best_confidence = 0.0
        self._best_prediction = None
    
    def should_guess(self, logits: torch.Tensor, elapsed_time: Optional[float] = None, 
                     **kwargs) -> Tuple[bool, Optional[int], Optional[float]]:
        num_strokes_seen = kwargs.get('num_strokes_seen')
        if num_strokes_seen is None:
            raise ValueError("TimeBasedPolicy requires num_strokes_seen in kwargs")
        
        if self._has_guessed:
            return False, None, None
        
        pred, conf = self._get_prediction(logits)
        
        if self.use_best_guess and conf > self._best_confidence:
            self._best_confidence = conf
            self._best_prediction = pred
        
        if num_strokes_seen >= self.num_strokes:
            self._has_guessed = True
            if self.use_best_guess and self._best_prediction is not None:
                return True, self._best_prediction, self._best_confidence
            return True, pred, conf
        
        return False, None, None
    
    def reset(self):
        self._has_guessed = False
        self._best_confidence = 0.0
        self._best_prediction = None


class LearnedPolicy(GuessingPolicy):
    """Learned guessing policy using a neural network."""
    
    def __init__(self, num_classes: int, threshold: float = 0.5, 
                 include_confidence: bool = True, include_time: bool = True, 
                 max_time: float = 30.0):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        self.num_classes = num_classes
        self.threshold = threshold
        self.include_confidence = include_confidence
        self.include_time = include_time
        self.max_time = max_time
        self._has_guessed = False
        
        input_dim = num_classes + int(include_confidence) + int(include_time)
        self.decision_layer = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.decision_layer.weight)
        nn.init.constant_(self.decision_layer.bias, -1.0)
    
    def should_guess(self, logits: torch.Tensor, elapsed_time: Optional[float] = None, 
                     **kwargs) -> Tuple[bool, Optional[int], Optional[float]]:
        if self._has_guessed:
            return False, None, None
        
        if self.include_time and elapsed_time is None:
            raise ValueError("LearnedPolicy with include_time=True requires elapsed_time")
        
        probs = F.softmax(logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        
        features = [probs]
        if self.include_confidence:
            features.append(torch.max(probs).unsqueeze(0))
        if self.include_time:
            features.append(torch.tensor([min(elapsed_time / self.max_time, 1.0)],
                                        dtype=probs.dtype, device=probs.device))
        
        with torch.no_grad():
            guess_prob = torch.sigmoid(self.decision_layer(torch.cat(features))).item()
        
        if guess_prob >= self.threshold:
            pred, conf = self._get_prediction(logits)
            self._has_guessed = True
            return True, pred, conf
        
        return False, None, None
    
    def reset(self):
        self._has_guessed = False
    
    def train_mode(self, mode: bool = True):
        self.decision_layer.train(mode)
    
    def eval_mode(self):
        self.decision_layer.eval()
    
    def get_parameters(self):
        return self.decision_layer.parameters()
    
    def load_state_dict(self, state_dict):
        self.decision_layer.load_state_dict(state_dict)
    
    def state_dict(self):
        return self.decision_layer.state_dict()


class PolicyWrapper:
    """Combines a classification model with a guessing policy."""
    
    def __init__(self, model: nn.Module, policy: GuessingPolicy, 
                 device: torch.device = torch.device('cpu')):
        self.model = model
        self.policy = policy
        self.device = device
        self.model.to(device).eval()
    
    def predict(self, image: torch.Tensor, elapsed_time: Optional[float] = None, 
                **kwargs) -> Tuple[bool, Optional[int], Optional[float], torch.Tensor]:
        """Returns (should_guess, predicted_class, confidence, logits)."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(image.to(self.device))
        
        should_guess, pred, conf = self.policy.should_guess(
            logits, elapsed_time=elapsed_time, **kwargs)
        
        return should_guess, pred, conf, logits
    
    def reset(self):
        self.policy.reset()
    
    def get_policy(self) -> GuessingPolicy:
        return self.policy
    
    def get_model(self) -> nn.Module:
        return self.model


def create_policy(policy_type: str, num_classes: int = 345, 
                  **kwargs) -> GuessingPolicy:
    """Factory function to create guessing policies."""
    policies = {
        'confidence': ConfidenceThresholdPolicy,
        'time': TimeBasedPolicy,
        'learned': lambda **kw: LearnedPolicy(num_classes=num_classes, **kw),
    }
    
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}. "
                        f"Available: {list(policies.keys())}")
    
    return policies[policy_type](**kwargs)
