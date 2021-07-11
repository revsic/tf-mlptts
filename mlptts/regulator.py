from typing import Tuple

import numpy as np
import tensorflow as tf


class PointwiseBlender(tf.keras.Model):
    """Nonlienar pointwise blender for positional information and context.
    """
    def __init__(self, channels: int, kernels: int, mlp: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernel.
            mlp: size of the mlp channels.
        """
        super().__init__()
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Conv1D(channels, kernels, padding='same'),
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Activation(tf.nn.silu)])
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp, activation=tf.nn.silu),
            tf.keras.layers.Dense(mlp, activation=tf.nn.silu)])
    
    def call(self, context: tf.Tensor, pos: tf.Tensor) -> tf.Tensor:
        """Blend the positional information.
        Args:
            context: [tf.float32; [B, S, C]], context.
            pos: [tf.float32; [B, T, S, P]], matrix representation of positional kernel.
        Returns:
            [tf.float32; [B, T, S, M]], blended matrix.
        """
        timestep = tf.shape(pos)[1]
        # [B, T, S, C]
        context = tf.tile(self.proj(context)[:, None], [1, timestep, 1, 1])
        # [B, T, S, M]
        return self.mlp(tf.concat([context, pos], axis=-1))


class Regulator(tf.keras.Model):
    """Length regulator introduced by Parallel Tacotron 2.
    """
    def __init__(self, channels: int, conv: int, kernels: int, mlp: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            conv: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            mlp: size of the mlp channels.
        """
        super().__init__()
        self.mlp_weight = PointwiseBlender(conv, kernels, mlp)
        self.proj_weight = tf.keras.layers.Dense(1)
        self.mlp_aux = PointwiseBlender(conv, kernels, mlp)
        self.proj_aux = tf.keras.layers.Dense(channels)
    
    def call(self, context: tf.Tensor, durations: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Align the context.
        Args:
            context: [tf.float32; [B, S, C]], input context.
            durations: [tf.float32; [B, S]], speech durations of each text tokens.
            mask: [tf.float32; [B, T, S]], attention mask.
        Returns:
            weights: [tf.float32; [B, T, S]], attention alignment.
            aligned: [tf.float32; [B, T, C]], upsampled tensor.
        """
        # [B, S]
        end = tf.math.cumsum(durations, axis=-1)
        # [B, S]
        start = end - durations

        # [1, T, 1]
        timestep = tf.range(tf.shape(mask)[1], dtype=tf.float32)[None, :, None]
        # [B, T, S]
        start = timestep - start[:, None]
        # [B, T, S]
        end = end[:, None] - start
        # [B, T, S, 2]
        pos = tf.concat([start[..., None], end[..., None]], axis=-1)

        # [B, T, S]
        weights = tf.squeeze(
            self.proj_weight(self.mlp_weight(context, pos)), axis=-1)
        # [B, T, S]
        weights = mask[:, 0:1] * weights + (1 - mask[:, 0:1]) * -1e6
        # [B, T, S]
        weights = tf.nn.softmax(weights, axis=-1) * mask
        # [B, T, C]
        aligned = tf.matmul(weights, context)

        # [B, T, M]
        aux = tf.reduce_sum(
            self.mlp_aux(context, pos) * weights[..., None], axis=2)
        # [B, T, C]
        aligned = self.proj_aux(tf.concat([aligned, aux], axis=-1))
        return weights, aligned
