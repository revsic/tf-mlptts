import tensorflow as tf

from .mlp import ChannelMLP, ConvMLP


class MLPMixer(tf.keras.Model):
    """MLP-Mixer variant for variable length inputs.
    """
    def __init__(self,
                 numlayers: int,
                 channels: int,
                 ch_hiddens: int,
                 kernels: int,
                 strides: int,
                 tp_hiddens: int,
                 dropout: float):
        """Initializer.
        Args:
            numlayers: the number of the mixer layers.
            channels: size of the input channels.
            ch_hiddens: size of the hidden channels.
            kernels: size of the frame.
            strides: step size of the next adjacent frame.
            tp_hiddens: size of the temporal hidden channels.
            dropout: dropout rate.
        """
        super().__init__()
        self.blocks = tf.keras.Sequential([
            tf.keras.Sequential([
                ChannelMLP(channels, ch_hiddens, dropout),
                ConvMLP(kernels, strides, tp_hiddens, dropout)])
            for _ in range(numlayers)])
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transform the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed inputs.
        """
        return self.blocks(inputs)
