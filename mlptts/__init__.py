from typing import Dict, Optional, Tuple

import tensorflow as tf

from .config import Config
from .mlpmixer import MLPMixer
from .regulator import Regulator


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
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.Embedding(config.vocabs, config.embedding),
            tf.keras.layers.Conv1D(config.channels, config.prenet_kernels, padding='same')])

        self.textenc = MLPMixer(
            config.text_layers,
            config.channels,
            config.text_ch_hiddens,
            config.text_kernels,
            config.text_strides,
            config.text_tp_hiddens,
            config.text_dropout)

        self.durator = tf.keras.Sequential([
            MLPMixer(
                config.dur_layers,
                config.channels,
                config.dur_ch_hiddens,
                config.dur_kernels,
                config.dur_strides,
                config.dur_tp_hiddens,
                config.dur_dropout),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation(tf.nn.softplus)])

        self.regulator = Regulator(
            config.channels, config.reg_conv, config.reg_kernels, config.reg_mlp)

        self.meldec = tf.keras.Sequential([
            MLPMixer(
                config.mel_layers,
                config.channels,
                config.mel_ch_hiddens,
                config.mel_kernels,
                config.mel_strides,
                config.mel_tp_hiddens,
                config.mel_dropout),
            tf.keras.layers.Dense(config.mel)])

    def call(self,
             text: tf.Tensor,
             textlen: tf.Tensor,
             mellen: Optional[tf.Tensor] = None) -> Tuple[
                 tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate log-mel scale power spectrogram from text tokens.
        Args:
            text: [tf.int32; [B, S]], text tokens.
            textlen: [tf.int32; [B]], length of the text sequence.
            mellen: [tf.int32; [B]], desired mel lengths.
        Returns:
            mel: [tf.float32; [B, T, mel]], log-mel scale power spectrogram.
            mellen: [tf.int32; [B]], length of the mel spectrogram.
            weights: [tf.float32; [B, T, S]], attention alignment.
            durations: [tf.float32; [B, S]], speech durations of each text tokens.
        """
        ## 1. Text encoding
        # [B, S]
        text_mask = self.mask(textlen, tf.shape(text)[1])
        # [B, S, C]
        embeddings = self.embedding(text) * text_mask[..., None]
        # [B, S, C]
        context = self.textenc(embeddings) * text_mask[..., None]
        
        ## 2. Inference duration
        # [B, S]
        durations = tf.squeeze(self.durator(context), axis=-1)
        # [B]
        inf_mellen = tf.reduce_sum(durations, axis=-1)
        if mellen is not None:
            # [B]
            factor = tf.cast(mellen, tf.float32) / inf_mellen
            # [B, S]
            durations = factor[:, None] * durations
        else:
            mellen = tf.cast(tf.math.ceil(inf_mellen), tf.int32)

        ## 3. Align
        # [B, T]
        mel_mask = self.mask(mellen)
        # [B, T, S]
        attn_mask = mel_mask[..., None] * text_mask[:, None]
        # [B, T, S], [B, T, C]
        weights, aligned = self.regulator(context, durations, attn_mask)

        ## 4. Decode mel-spectrogram.
        # [B, T, mel]
        mel = self.meldec(aligned) * mel_mask[..., None]
        return mel, mellen, weights, durations

    def compute_loss(self,
                     text: tf.Tensor,
                     textlen: tf.Tensor,
                     mel: tf.Tensor,
                     mellen: tf.Tensor) -> Tuple[
            tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
        """Compute loss of mlp-tts.
        Args:
            text: [tf.int32; [B, S]], text tokens.
            textlen: [tf.int32; [B]], length of the text sequence.
            mel: [tf.float32; [B, T, mel]], ground-truth mel-spectrogram.
            mellen: [tf.int32; [B]], length of the mel-spectrogram.
        Returns:
            loss: [tf.float32; []], loss values.
            losses: {key: [tf.float32; []]}, individual loss values.
            weights: [tf.float32; [B, T, S]], attention alignment. 
        """
        # [B, T, num_mels], _, [B, T, S], [B, S]
        inf_mel, _, weights, durations = self.call(text, textlen, mellen=mellen)
        # [B]
        mellen = tf.cast(mellen, tf.float32)
        # [B], l1-loss
        melloss = tf.reduce_sum(tf.abs(inf_mel - mel), axis=[1, 2])
        # []
        melloss = tf.reduce_mean(melloss / (mellen * self.config.mel))
        # []
        durloss = tf.reduce_mean(
            tf.abs(tf.math.log(tf.reduce_sum(durations)) - tf.math.log(mellen)))
        # []
        loss = melloss + durloss
        losses = {'melloss': melloss, 'durloss': durloss}
        return loss, losses, weights

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
