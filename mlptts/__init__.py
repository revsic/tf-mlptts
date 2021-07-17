from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import Config
from .mlpmixer import MLPMixer
from .pe import PositionalEncodings
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

        self.pe = PositionalEncodings(config.channels)
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
        self.proj_mu = tf.keras.layers.Dense(config.latent_channels)
        self.proj_sigma = tf.keras.layers.Dense(
            config.latent_channels, activation=tf.nn.softplus)

        self.proj_latent = tf.keras.layers.Dense(config.channels)

        self.durator = tf.keras.Sequential([
            MLPMixer(
                config.dur_layers,
                config.channels,
                config.dur_hiddens,
                config.dur_kernels,
                config.eps),
            tf.keras.layers.Dense(2, activation=tf.nn.softplus)])

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
                  attn: [tf.float32; [B, T, S]], attention alignment.
                  dursum: [tf.float32; [B]], sum of the speech durations.
                  mu: [tf.float32; [B, S, R]], latent mean.
                  sigma: [tf.float32; [B, S, R]], latent stddev.
                  latent: [tf.float32; [B, S, R]], latent variable.
        """
        ## 1. Text encoding
        seqlen = tf.shape(text)[1]
        # [B, S]
        text_mask = self.mask(textlen, seqlen)
        # [B, S, C]
        embeddings = (self.embedding(text) + self.pe(seqlen)) * text_mask[..., None]
        # [B, S, C]
        context = self.textenc(embeddings) * text_mask[..., None]

        ## 2. Residual encoding
        if mel is not None:
            # [B, T]
            mel_mask = self.mask(mellen, maxlen=tf.shape(mel)[1])
            # [B, T, C]
            residual = self.resenc(mel) * mel_mask[..., None]
            # [B, S, T]
            refattn_mask = text_mask[..., None] * mel_mask[:, None]
            # [B, S, C]
            aligned = self.refattn(context, residual, refattn_mask)
            # [B, S, C]
            mu, sigma = self.proj_mu(aligned), self.proj_sigma(aligned)
        else:
            mu, sigma = 0., 1.
        # [B, S, C]
        latent = tf.random.normal(
            [*tf.shape(context)[:2], self.config.latent_channels]) * sigma + mu
        # [B, S, C]
        context = self.proj_latent(tf.concat([context, latent], axis=-1))

        ## 3. Inference duration
        # [B, S, 1], [B, S, 1]
        dur, std = tf.split(self.durator(context), 2, axis=-1)
        # [B, S], [B, S]
        dur, std = tf.squeeze(dur, axis=-1), tf.squeeze(std, axis=-1)
        # [B]
        dursum = tf.reduce_sum(dur, axis=-1)
        if mellen is None:
            # [B]
            mellen = tf.cast(tf.math.ceil(dursum), tf.float32)
        else:
            # [B]
            factor = tf.cast(mellen, tf.float32) / dursum
            # [B, S]
            dur = factor[:, None] * dur

        ## 4. Align
        # [B, T]
        mel_mask = self.mask(mellen)
        # [B, T, S]
        attn_mask = mel_mask[..., None] * text_mask[:, None]
        # [B, T, S]
        weights = self.gaussian_upsampler(dur, std, attn_mask)
        # [B, T, C]
        aligned = tf.matmul(weights, context)

        ## 5. Decode mel-spectrogram.
        # [B, T, mel]
        mel = self.meldec(aligned) * mel_mask[..., None]
        # auxiliary features
        return mel, mellen, {
            'attn': weights, 'dursum': dursum, 'mu': mu, 'sigma': sigma, 'latent': latent}

    def compute_loss(self,
                     text: tf.Tensor,
                     textlen: tf.Tensor,
                     mel: tf.Tensor,
                     mellen: tf.Tensor) -> Tuple[
            tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Compute loss of mlp-tts.
        Args:
            text: [tf.int32; [B, S]], text tokens.
            textlen: [tf.int32; [B]], length of the text sequence.
            mel: [tf.float32; [B, T, mel]], ground-truth mel-spectrogram.
            mellen: [tf.int32; [B]], length of the mel-spectrogram.
        Returns:
            loss: [tf.float32; []], loss values.
            losses: {key: [tf.float32; []]}, individual loss values.
            aux: {key: tf.Tensor}, auxiliary outputs.
                  attn: [tf.float32; [B, T, S]], attention alignment.
                  mel: [tf.float32; [B, T, mel]], generated mel.
                  mellen: [tf.float32; [B]], length of the mel-spectrogram. 
        """
        # [B, T, mel], _, [B, T, S], _
        inf_mel, _, aux = self.call(text, textlen, mel=mel, mellen=mellen)
        # [B], [B]
        mellen, textlen = tf.cast(mellen, tf.float32), tf.cast(textlen, tf.float32)
        # [], l1-loss
        melloss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(inf_mel - mel), axis=1) / mellen[:, None])
        # []
        durloss = tf.reduce_mean(
            tf.square(tf.math.log(aux['dursum']) - tf.math.log(mellen)) / textlen)
        # [B, S, C]
        dkl = self.gll(aux['latent'], aux['mu'], aux['sigma']) - self.gll(aux['latent'])
        # []
        dkl = tf.reduce_mean(tf.reduce_sum(dkl, axis=1) / textlen[:, None])
        # []
        loss = melloss + durloss + dkl
        losses = {'melloss': melloss, 'durloss': durloss, 'dkl': dkl}
        return loss, losses, {
            'attn': aux['attn'], 'mel': inf_mel, 'mellen': tf.cast(mellen, tf.int32)}

    def gll(self, inputs: tf.Tensor, mean: tf.Tensor = 0., stddev: tf.Tensor = 1.) -> tf.Tensor:
        """Gaussian log-likelihood.
        Args:
            inputs: [tf.float32; [...]], input tensor.
            mean: [tf.float32; [...]], mean.
            stddev: [tf.float32; [...]], standard deviation.
        Returns:
            [tf.float32; [...]], likelihood.
        """
        # [...]
        logstd = tf.math.log(tf.maximum(stddev, 1e-5))
        return -0.5 * (np.log(2 * np.pi) + 2 * logstd +
            tf.exp(-2 * logstd) * tf.square(inputs - mean))

    def align_penalty(self, textlen: tf.Tensor, mellen: tf.Tensor, sigma: float = 0.3) -> tf.Tensor:
        """Penalty matrix for attention diagonality.
        Args:
            textlen: [tf.float32; [B]], lengths of the text sequences.
            mellen: [tf.float32; [B]], lengths of the mel spectrograms.
            sigma: penalty smoothness factor.
        Returns:
            [tf.float32; [B, T, S]], penalty matrix.
        """
        seqlen, timestep = tf.reduce_max(textlen), tf.reduce_max(mellen)
        # [S], [T]
        srange, trange = tf.range(seqlen, dtype=tf.float32), tf.range(timestep, dtype=tf.float32)
        # [B, S], [B, T]
        srange, trange = srange[None] / textlen[:, None], trange[None] / mellen[:, None]
        # [B, T, S]
        penalty = 1 - tf.math.exp(
            -tf.square(srange[:, None] - trange[..., None]) / (2 * sigma ** 2))
        # [B, T, S]
        mask = self.mask(textlen, seqlen)[:, None] * self.mask(mellen, timestep)[..., None]
        return penalty * mask

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

    def gaussian_upsampler(self, dur: tf.Tensor, std: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Generate alignment from duration and standard deviation.
        Args;
            dur: [tf.float32; [B, S]], speech duration for each text tokens.
            std: [tf.float32; [B, S]], standard deviation.
            mask: [tf.float32; [B, T, S]], attention mask.
        Returns:
            [tf.float32; [B, T, S]], attention alignment.
        """
        # [B, S]
        middle = tf.math.cumsum(dur, axis=-1) - 0.5 * dur
        # [1, T, 1]
        trange = tf.range(tf.shape(mask)[1], dtype=tf.float32)[None, :, None]
        # [B, T, S]
        logit = -tf.square(trange - middle[:, None]) / (2 * tf.square(std[:, None]))
        # [B, T, S]
        return tf.nn.softmax(logit * mask[..., 0:1], axis=-1) * mask

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
