from typing import Dict, Tuple

import tensorflow as tf

from mlptts import MLPTextToSpeech


class TrainWrapper:
    """MLP-TTS train wrapper.
    """
    def __init__(self, model: MLPTextToSpeech):
        """Initializer.
        Args:
            model: target model.
        """
        self.model = model

    def forward(self, *args, **kwargs):
        """Forward to the model.
        """
        return self.model(*args, **kwargs)

    def apply_gradient(self,
                       optim: tf.keras.optimizers.Optimizer,
                       text: tf.Tensor,
                       textlen: tf.Tensor,
                       mel: tf.Tensor,
                       mellen: tf.Tensor) -> Tuple[
            tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Apply gradients.
        Args:
            optim: optimizer.
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
                  mellen: [tf.float32; [B]], length of the mel-spectrogram
                  gradnorm: gradient norm.
                  paramnorm: parameter norm.
        """
        with tf.GradientTape() as tape:
            loss, losses, aux = self.compute_loss(text, textlen, mel, mellen)
        # compute gradient
        grad = tape.gradient(loss, self.model.trainable_variables)
        # update
        optim.apply_gradients(zip(grad, self.model.trainable_variables))
        # norms
        gradnorm = tf.reduce_mean([
            tf.norm(g) for g in grad if g is not None])
        paramnorm = tf.reduce_mean([
            tf.norm(p) for p in self.model.trainable_variables])
        return loss, losses, {
            'gradnorm': gradnorm.numpy().item(),
            'paramnorm': paramnorm.numpy().item(), **aux}

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
        pred, _, aux = self.model(text, textlen, mel=mel, mellen=mellen)
        # [B, S]
        text_mask = aux['mask'][:, 0]

        ## 1. Mel-spectrogram prediction loss
        # [B], [B]
        mellen, textlen = tf.cast(mellen, tf.float32), tf.cast(textlen, tf.float32)
        # [], l1-loss
        melloss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(pred - mel), axis=1) / mellen[:, None])

        ## 2. Duration loss
        # [B, S]
        logdur = tf.math.log(tf.maximum(tf.reduce_sum(aux['attn'], axis=1), 1.))
        # [B, S]
        predur = tf.math.log(tf.maximum(aux['durations'], 1.))
        # []
        durloss = tf.reduce_mean(
            tf.reduce_sum(tf.square(logdur - predur), axis=1) / textlen)
        # []
        ctcloss = -tf.reduce_mean(aux['ctc'])

        ## 3. Residual latent
        # [B, S, C]
        dkl = self.nll(aux['latent']) - self.nll(aux['latent'], aux['mu'], aux['sigma'])
        # []
        dkl = tf.reduce_mean(
            tf.reduce_sum(dkl * text_mask[..., None], axis=1) / textlen[:, None])
        # []
        loss = melloss + self.model.config.ctc_lambda * ctcloss + durloss + dkl
        losses = {'melloss': melloss, 'ctcloss': ctcloss, 'durloss': durloss, 'dkl': dkl}
        return loss, losses, {
            'attn': aux['attn'], 'mel': pred, 'mellen': tf.cast(mellen, tf.int32)}

    def nll(self, inputs: tf.Tensor, mean: tf.Tensor = 0., stddev: tf.Tensor = 1.) -> tf.Tensor:
        """Gaussian negative log-likelihood.
        Args:
            inputs: [tf.float32; [...]], input tensor.
            mean: [tf.float32; [...]], mean.
            stddev: [tf.float32; [...]], standard deviation.
        Returns:
            [tf.float32; [...]], likelihood.
        """
        logstd = tf.math.log(tf.maximum(stddev, 1e-5))
        return 2 * logstd + tf.exp(-2 * logstd) * tf.square(inputs - mean)
