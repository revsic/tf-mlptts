import tensorflow as tf


class ChannelMLP(tf.keras.Model):
    """MLP-layer introduced by MLP-Mixer.
    """
    def __init__(self, channels: int, hiddens: int, dropout: float = 0.):
        """Initializer.
        Args:
            channels: size of the input channels.
            hiddens: size of the hidden channels.
            dropout: dropout rate.
        """
        super().__init__()
        self.transform = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Dense(hiddens, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(channels),
            tf.keras.layers.Dropout(dropout)])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """MLP transform.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed tensor.
        """
        return inputs + self.transform(inputs)


class ConvMLP(tf.keras.Model):
    """Convolution based temporal MLP for variable-length inputs.
    """
    def __init__(self,
                 kernels: int,
                 strides: int,
                 hiddens: int,
                 dropout: float = 0.):
        """Initializer.
        Args:
            kernels: size of the receptive field.
            strides: step size to the next adjacent frame.
            dropout: dropout rate.
        """
        super().__init__()
        self.strides = strides
        self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        self.framed_mlp = tf.keras.Sequential([
            # [B, T, C, 1] => [B, T // strides, C, H]
            # same as `Frame K-size, S-step -> Linear KxH -> GELU`.
            tf.keras.layers.Conv2D(
                hiddens, (kernels, 1), (strides, 1),
                padding='same', activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            # [B, T // strides, C, H] -> [B, T, C, 1]
            # same as `Linear HxK -> Overlap-and-Add K-size, S-step`
            tf.keras.layers.Conv2DTranspose(
                1, (kernels, 1), (strides, 1), padding='same'),
            tf.keras.layers.Dropout(dropout)])
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Overlap based variable-length MLP.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed tensor.
        """
        timestep = tf.shape(inputs)[1]
        # [B, T, C]
        x = self.layernorm(inputs)
        # [B, T + P, C]
        x = tf.squeeze(self.framed_mlp(x[..., None]), axis=-1)
        # P // 2
        padsize = (tf.shape(x)[1] - timestep) // 2
        # [B, T, C]
        return inputs + x[:, padsize:padsize + timestep]
