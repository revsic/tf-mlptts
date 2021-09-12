from typing import Optional, Tuple

import tensorflow as tf

from .aligner import Aligner
from .config import Config
from .mlpmixer import MLPMixer
from .refattn import ReferenceAttention


class MLPTextToSpeech(tf.keras.Model):
    """MLP-Mixer based TTS system, from text to log-mel scale power spectrogram.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configuration.
        """
        super().__init__()
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocabs, config.channels)

        self.textenc = MLPMixer(
            config.text_layers,
            config.channels,
            config.text_hiddens,
            config.text_kernels,
            config.eps)

        self.resenc = MLPMixer(
            config.res_layers,
            config.res_channels,
            config.res_hiddens,
            config.res_kernels,
            config.eps)

        self.refattn = ReferenceAttention(config.res_channels)
        self.proj_var = tf.keras.layers.Dense(config.latent_channels * 2)
        self.proj_latent = tf.keras.layers.Dense(config.channels)

        self.aligner = Aligner(
            config.mel,
            config.channels,
            config.ctc_hiddens,
            config.ctc_layers,
            config.ctc_factor,
            config.eps)

        self.durator = tf.keras.Sequential([
            MLPMixer(
                config.dur_layers,
                config.channels,
                config.dur_hiddens,
                config.dur_kernels,
                config.eps),
            tf.keras.layers.Dense(1, activation=tf.nn.softplus)])

        self.meldec = tf.keras.Sequential([
            MLPMixer(
                config.mel_layers,
                config.channels,
                config.mel_hiddens,
                config.mel_kernels,
                config.eps),
            tf.keras.layers.Dense(config.mel)])

    def call(self,
             text: tf.Tensor,
             textlen: tf.Tensor,
             mel: Optional[tf.Tensor] = None,
             mellen: Optional[tf.Tensor] = None) -> Tuple[
                 tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate log-mel scale power spectrogram from text tokens.
        Args:
            text: [tf.int32; [B, S]], text tokens.
            textlen: [tf.int32; [B]], length of the text sequence.
            mel: [tf.float32; [B, T, mel]], reference mel-spectrogram.
            mellen: [tf.int32; [B]], desired mel lengths.
        Returns:
            mel: [tf.float32; [B, T, mel]], log-mel scale power spectrogram.
            mellen: [tf.int32; [B]], length of the mel spectrogram.
            aux: {key: tf.Tensor}, auxiliary features.
        """
        ## 1. Text encoding
        bsize, seqlen = tf.shape(text)
        # [B, S]
        text_mask = self.mask(textlen, seqlen)
        # [B, S, C]
        embeddings = self.embedding(text) * text_mask[..., None]
        # [B, S, C]
        context = self.textenc(embeddings) * text_mask[..., None]

        ## 2. Residual encoding
        if mel is not None:
            # [B, T]
            mel_mask = self.mask(mellen, maxlen=tf.shape(mel)[1])
            # [B, T, S]
            attn_mask = text_mask[:, None] * mel_mask[..., None]
            # [B, T, C]
            residual = self.resenc(mel) * mel_mask[..., None]
            # [B, S, C]
            aligned = self.refattn(context, residual, tf.transpose(attn_mask, [0, 2, 1]))
            # [B, S, C]
            mu, sigma = tf.split(self.proj_var(aligned), 2, axis=-1)
        else:
            mu, sigma = 0., 1.
        # [B, S, C]
        latent = tf.random.normal(
            [bsize, seqlen, self.config.latent_channels]) * sigma + mu
        # [B, S, C]
        context = self.proj_latent(tf.concat([context, latent], axis=-1))

        ## 3. CTC-like alignment search
        # [B, S]
        durations = tf.squeeze(self.durator(context), axis=-1) * text_mask
        if mel is not None:
            # [B, T // F, S], [B, T, S]
            log_prob, attn = self.aligner(context, mel, attn_mask)
        else:
            # placeholders
            log_prob = None
            # [B, S], quantize
            durations = tf.cast(tf.round(durations), tf.int32)
            # [B]
            mellen = tf.reduce_sum(durations, axis=-1)
            # [B, T]
            mel_mask = self.mask(mellen)
            # [B, T, S]
            attn_mask = text_mask[:, None] * mel_mask[..., None]
            # [B, T, S]
            attn = self.attention(durations, attn_mask)
        # [B, T, C]
        aligned = tf.matmul(attn, context)

        ## 5. Decode mel-spectrogram.
        # [B, T, mel]
        mel = self.meldec(aligned) * mel_mask[..., None]
        # auxiliary features
        return mel, mellen, {
            'durations': durations, 'attn': attn, 'mask': attn_mask,
            # residual latent
            'mu': mu, 'sigma': sigma, 'latent': latent,
            # ctc operation
            'log_prob': log_prob}

    def mask(self, lengths: tf.Tensor, maxlen: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Generate the mask from length vectors.
        Args:
            lengths: [tf.int32; [B]], length vector.
            maxlen: [tf.int32; []], output time steps.
        Returns:
            [tf.float32; [B, maxlen]], binary mask.
        """
        if maxlen is None:
            maxlen = tf.reduce_max(lengths)
        # [B, maxlen]
        return tf.cast(tf.range(maxlen)[None] < lengths[:, None], tf.float32)

    def attention(self, durations: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate attention alignment from durations.
        Args:
            durations: [tf.float32; [B, S]], duration vectors.
            mask: [tf.float32; [B, S]], alignment mask.
        Returns:
            align: [tf.float32; [B, T, S]], attention alignment.
        """
        # B, T, S
        bsize, timestep, seqlen = tf.shape(mask)
        # [B x S]
        cumsum = tf.reshape(tf.math.cumsum(durations, axis=-1), [-1])
        # [B, S, T]
        cumattn = tf.reshape(self.mask(cumsum, timestep), [bsize, seqlen, timestep])
        # [B, S, T]
        attn = cumattn - tf.pad(cumattn[:, :-1], [[0, 0], [1, 0], [0, 0]])
        # [B, T, S]
        return tf.transpose(attn, [0, 2, 1]) * mask

    def write(self, path: str,
              optim: Optional[tf.keras.optimizers.Optimizer] = None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: path to write.
            optim: optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path: str,
                optim: Optional[tf.keras.optimizers.Optimizer] = None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: path to restore.
            optim: optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        return ckpt.restore(path)
