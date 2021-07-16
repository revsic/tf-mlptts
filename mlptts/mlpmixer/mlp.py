from typing import List

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
            tf.keras.layers.Dense(hiddens, activation=tf.nn.swish),
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
            # same as `Frame K-size, S-step -> Linear KxH -> Swish`.
            tf.keras.layers.Conv2D(
                hiddens, (kernels, 1), (strides, 1),
                padding='same', activation=tf.nn.swish),
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


class DynWeightMLP(tf.keras.Model):
    """MLP with dynamic weights, computed by `proj(concat(inputs, inputs.T))`.
    WARNING: currently, modification of temporal scale is impossible.
    """
    def __init__(self, eps: float = 1e-3):
        """Initializer.
        Args:
            eps: small value for layer scale.
        """
        super().__init__()
        self.proj_upper = tf.keras.layers.Dense(1)
        self.proj_lower = tf.keras.layers.Dense(1)
        self.proj_bias = tf.keras.layers.Dense(1)
        self.scale = tf.Variable(eps)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute weights and project.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T x F, C]], projected input.
        """
        # [B, T, 1], [B, T, 1]
        upper, lower = self.proj_upper(inputs), self.proj_lower(inputs)
        # [B, T, C], optimized
        weighted = self.scale * (
            # [B, T, 1] * [B, 1, C] + [B, 1, T] x [B, T, C]
            upper * tf.reduce_sum(inputs, axis=1)[:, None]
            + tf.matmul(tf.transpose(lower, [0, 2, 1]), inputs))
        return weighted + self.proj_bias(inputs)


class TemporalConv(tf.keras.Model):
    """Convolution only on temporal axis.
    """
    def __init__(self, kernels: int, dilations: int, dropout: float = 0.):
        """Initializer.
        Args:
            kernels: size of the convolutional kernels.
            dilations: dilation rate.
            dropout: dropout rate.
        """
        super().__init__()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.transform = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                1, (kernels, 1), padding='same', dilation_rate=(dilations, 1)),
            tf.keras.layers.Activation(tf.nn.swish),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(
                1, (kernels, 1), padding='same', dilation_rate=(dilations, 1))])
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transform the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed.
        """
        # [B, T, C]
        x = self.layernorm(inputs)
        # [B, T, C]
        return inputs + tf.squeeze(self.transform(x[..., None]), axis=-1)


class DynTemporalMLP(tf.keras.Model):
    """Temporal MLP with dynamic weights.
    """
    def __init__(self, eps: float = 1e-3, dropout: float = 0.):
        """Initializer.
        Args:
            eps: small value for layer scale.
            dropout: dropout rate.
        """
        super().__init__()
        self.transform = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(axis=-1),
            DynWeightMLP(eps),
            tf.keras.layers.Activation(tf.nn.swish),
            tf.keras.layers.Dropout(dropout),
            DynWeightMLP(eps),
            tf.keras.layers.Dropout(dropout)])
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """MLP transform.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed tensor.
        """
        return inputs + self.transform(inputs)


class ResBlock(tf.keras.Model):
    """Residual block.
    """
    def __init__(self, num_layers: int, channels: int, kernels: int, dilations: int):
        """Initializer.
        Args:
            num_layers: the number of the convolutional blocks
                before residual connection.
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dilations: dilation rate.
        """
        super().__init__()
        self.blocks = tf.keras.Sequential([
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    channels, kernels, padding='same',
                    dilation_rate=dilations, activation='relu'),
                tf.keras.layers.BatchNormalization(axis=-1)])
            for _ in range(num_layers)])
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transform inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed.
        """
        return inputs + self.blocks(inputs)


class ResNet(tf.keras.Model):
    """Residual network for POC.
    """
    def __init__(self, num_layers: int, channels: int,
                 kernels: int, dilations: List[int]):
        """Initializer.
        Args:
            num_layers: the number of the convolutional blocks
                before residual connection.
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dilations: dilation rates.
        """
        super().__init__()
        self.blocks = tf.keras.Sequential([
            ResBlock(num_layers, channels, kernels, dilation)
            for dilation in dilations])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transform inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T, C]], transformed.
        """
        return self.blocks(inputs)
