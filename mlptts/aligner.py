from typing import Optional, Tuple

import tensorflow as tf

from .mlpmixer import ChannelMLP


class Aligner(tf.keras.Model):
    """CTC-inspired aligner
    """
    def __init__(self,
                 mel: int,
                 channels: int,
                 hiddens: int,
                 layers: int,
                 factor: int,
                 eps: float = 1e-3):
        """Initializer.
        Args:
            reduction: reduction factor.
        """
        super().__init__()
        self.mel = mel
        self.factor = factor
        self.mdn = tf.keras.Sequential([
            ChannelMLP(channels, hiddens, eps)
            for _ in range(layers)])
        self.mdn.add(tf.keras.layers.Dense(mel * factor * 2))

    def call(self, context: tf.Tensor, mel: tf.Tensor, mask: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        """Monotonic-alignment search.
        Args:
            context: [tf.float23; [B, S, C]], text encodings.
            mel: [tf.float32; [B, T, mel]], ground-truh mel-spectrogram.
            mask: [tf.float32; [B, T, S]], attention mask.
        Returns:
            log_prob: [tf.float32; [B, T, S]], log-probability.
            align: [tf.float32; [B, T, S]], attention alignment.
        """
        # B, T, S
        bsize, timestep, seqlen = tf.shape(mask)
        # [B, T // F, F x mel]
        mel, remain = self.reduction(mel)
        # [B, T // F, F x S]
        rmask, _ = self.reduction(mask)
        # [B, T // F, S]
        rmask = tf.reduce_sum(tf.reshape(rmask, [bsize, -1, self.factor, seqlen]), axis=2)
        rmask = tf.minimum(rmask, 1.)
        # [B, S, F x mel], [B, S, F x mel]
        mu, logs = tf.split(self.mdn(context), 2, axis=-1)
        # [B, T // F, S]
        log_prob = -2 * tf.reduce_mean(logs, axis=-1)[:, None] - (
            tf.reduce_sum(tf.square(mu * tf.exp(-logs)), axis=-1)[:, None]
            -2 * tf.matmul(mel, tf.transpose(mu * tf.exp(-2 * logs), [0, 2, 1]))
            + tf.matmul(tf.square(mel), tf.transpose(tf.exp(-2 * logs), [0, 2, 1]))) / self.mel * rmask
        # [B, T // F, S]
        align = self.search(log_prob, rmask)
        # [B, T // F, F, S]
        align = tf.tile(align[..., None, :], [1, 1, self.factor, 1])
        # [B, T, S]
        align = tf.reshape(align, [bsize, -1, seqlen])
        if remain is not None:
            # [B, T, S]
            align = align[:, :-remain]
        # [B, T // F, S], [B, T, S]
        return log_prob, tf.stop_gradient(align)

    def search(self, log_prob: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Monotonic-alignment search.
        Args:
            log_prob: [tf.float32; [B, T, S]], log-probability.
            mask: [tf.float32; [B, T, S]], attenion mask.
        Returns:
            [tf.float32; [B, T, S]], alignment.
        """
         # B, T, S
        bsize, timestep, seqlen = tf.shape(log_prob)
        # [B, S]
        prob = tf.zeros([bsize, seqlen], dtype=tf.float32)
        # [1, S]
        arange = tf.range(seqlen)[None]
        # T x [B, S]
        dirs = []
        for j in tf.range(timestep):
            # [B, S]
            prev = tf.pad(prob, [[0, 0], [1, 0]], constant_values=tf.float32.min)[:, :-1]
            # [B, S]
            dirs.append(prob >= prev)
            # [B, S]
            prob = tf.where(arange <= j, tf.maximum(prob, prev) + log_prob[:, j], tf.float32.min)
        # [B, T, S]
        dirs = tf.cast(tf.stack(dirs, axis=1), tf.int32)
        # [B, T, S]
        dirs = tf.where(tf.cast(mask, tf.bool), dirs, 1)
        # [B]
        index = tf.cast(tf.reduce_sum(mask[:, 0], axis=-1), tf.int32) - 1
        # [B], [B]
        batch, ones = tf.range(bsize), tf.ones(bsize)
        # T x [B, S]
        attn = []
        for j in tf.range(timestep)[::-1]:
            # [B, S]
            attn.append(tf.scatter_nd(
                tf.stack([batch, index], axis=1), ones, [bsize, seqlen]))
            # [B], gather-nd = dirs[batch, j, index]
            index = index - 1 + tf.gather_nd(
                dirs, tf.stack([batch, tf.fill([bsize], j), index], axis=1))
        # [B, T, S]
        return tf.stack(attn[::-1], axis=1) * mask

    def marginalize(self, log_prob: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Marginalize the log-probability.
        Args:
            log_prob: [tf.float32; [B, T, S]], log-probability matrix.
            mask: [tf.float32; [B, T, S]], alignment mask.
        Returns:
            [tf.float32; [B]], marginalized likelihood.
        """
        # B, S
        bsize, _, seqlen = tf.shape(log_prob)
        # [B, S - 1]
        log_alpha = tf.fill([bsize, seqlen - 1], tf.float32.min)
        # [B, S]
        log_alpha = tf.concat([log_prob[:, 0, 0:1], log_alpha], axis=-1)
        # T x [B, S]
        log_alphas = [log_alpha]
        # (T - 1) x [B, S]
        for score in tf.transpose(log_prob, [1, 0, 2])[1:]:
            # [B, S]
            prev = tf.pad(log_alpha[:, :-1], [[0, 0], [1, 0]], constant_values=tf.float32.min)
            # [B, S]
            log_alpha = tf.reduce_logsumexp(tf.stack([log_alpha, prev]), axis=0) + score
            log_alphas.append(log_alpha)
        # [B, T, S]
        log_alphas = tf.stack(log_alphas, axis=1)
        # [B]
        textlen = tf.cast(tf.reduce_sum(mask[:, 0], axis=-1), tf.int32)
        mellen = tf.cast(tf.reduce_sum(mask[..., 0], axis=-1), tf.int32)
        # [B]
        return tf.gather_nd(
            log_alphas, tf.stack([tf.range(bsize), mellen - 1, textlen - 1], axis=-1))

    def reduction(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Reduce the timesteps.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
        Returns:
            [tf.float32; [B, T // F, F x C]], reduced tensor.
        """
        # B, T, C
        bsize, timestep, channels = tf.shape(inputs)
        if timestep % self.factor > 0:
            # R
            remain = self.factor - timestep % self.factor
            # [B, T + R, C]
            inputs = tf.pad(inputs, [[0, 0], [0, remain], [0, 0]])
        else:
            remain = None
        # [B, T // F, F x C]
        return tf.reshape(inputs, [bsize, -1, self.factor * channels]), remain
