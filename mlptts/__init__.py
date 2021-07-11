from typing import Optional, Tuple

import tensorflow as tf

from speechset import AcousticDataset

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
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                AcousticDataset.VOCABS, config.embedding),
            tf.keras.layers.Conv1D(config.channels, config.prenet_kernels, padding='same'),
            tf.keras.layers.Activation(tf.nn.silu)])

        self.textenc = MLPMixer(
            config.text_layers,
            config.channels,
            config.text_ch_hiddens,
            config.text_kernels,
            config.text_strides,
            config.text_tp_hiddens,
            config.text_dropout)

        self.durator = tf.keras.Sequential([
            tf.keras.layers.Dense(config.dur_channels),
            tf.keras.Sequential([
                MLPMixer(1, config.dur_channels, config.dur_channels,
                         kernels, 1, 1, dropout=config.dur_dropout)
                for kernels in config.dur_kernels]),
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
        mel = self.meldec(aligned)
        return mel, mellen, weights, durations

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