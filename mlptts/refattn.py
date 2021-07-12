import tensorflow as tf


class ReferenceAttention(tf.keras.Model):
    """Align residual signal.
    """
    def __init__(self, channels: int):
        """Initializer.
        Args:
            channels: size of the input channels.
        """
        super().__init__()
        self.scale = channels ** -0.5
        self.proj_key = tf.keras.layers.Dense(channels)
        self.proj_query = tf.keras.layers.Dense(channels)
        self.proj_value = tf.keras.layers.Dense(channels)

    def call(self, context: tf.Tensor, residual: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Align the residual signal with respect to the context query.
        Args:
            context: [tf.float32; [B, S, C]], context query.
            residual: [tf.float32; [B, T, C]], residual signal.
            mask: [tf.float32; [B, S, T]], attention mask.
        Returns:
            [tf.float32; [B, S, C]], aligned residual signal.
        """
        # [B, S, C]
        query = self.proj_query(context)
        # [B, T, C]
        key, value = self.proj_key(residual), self.proj_value(residual)
        # [B, S, T]
        weights = tf.matmul(query, tf.transpose(key, [0, 2, 1])) * self.scale
        # timestep masking
        weights = weights * mask[:, 0:1] + (1 - mask[:, 0:1]) * -1e6
        # [B, S, T]
        weights = tf.nn.softmax(weights, axis=-1) * mask
        # [B, S, C]
        return tf.matmul(weights, value)
