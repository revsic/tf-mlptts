import tensorflow as tf

from .mlp import ChannelMLP, DynTemporalMLP


class MLPMixer(tf.keras.Model):
    """MLP-Mixer variant for variable length inputs.
    """
    def __init__(self,
                 numlayers: int,
                 channels: int,
                 ch_hiddens: int,
                 eps: float,
                 dropout: float = 0.):
        """Initializer.
        Args:
            numlayers: the number of the mixer layers.
            channels: size of the input channels.
            ch_hiddens: size of the hidden channels.
            eps: small value for layer scale in dyn-temporal mlp.
            dropout: dropout rate.
        """
        super().__init__()
        self.blocks = tf.keras.Sequential([
            tf.keras.Sequential([
                ChannelMLP(channels, ch_hiddens, dropout),
                DynTemporalMLP(eps, dropout)])
            for _ in range(numlayers)])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transform the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed inputs.
        """
        return self.blocks(inputs)
