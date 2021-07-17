import tensorflow as tf

from .mlp import ChannelMLP, TemporalConv


class MLPMixer(tf.keras.Model):
    """MLP-Mixer variant for variable length inputs.
    """
    def __init__(self,
                 numlayers: int,
                 channels: int,
                 hiddens: int,
                 kernels: int,
                 eps: float = 1e-3):
        """Initializer.
        Args:
            numlayers: the number of the mixer layers.
            channels: size of the input channels.
            hiddens: size of the hidden channels.
            kernels: size of the convolutional kernels.
            eps: small value for pre-affine scaler.
        """
        super().__init__()
        self.blocks = tf.keras.Sequential([
            tf.keras.Sequential([
                ChannelMLP(channels, hiddens, eps),
                TemporalConv(channels, kernels, 1, eps)])
            for _ in range(numlayers)])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transform the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed inputs.
        """
        return self.blocks(inputs)
