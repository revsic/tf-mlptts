from typing import Any, Dict

import tensorflow as tf


class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup the learning rate.
    """
    def __init__(self, learning_rate: float, warmup_steps: int, alpha: float):
        """Initializer.
        Args:
            learning_rate: target learning rates.
            warmup_steps: the number of the warmup steps.
            alpha: 
        """
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.alpha = alpha
    
    def __call__(self, step: int) -> float:
        """Compute learning rates.
        Args:
            step: training steps.
        Returns:
            learning rates.
        """
        return self.learning_rate * tf.minimum(
            tf.exp((step - self.warmup_steps) / self.warmup_steps * self.alpha), 1.)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration.
        """
        return {
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
        }
