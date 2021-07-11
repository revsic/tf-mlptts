class Config:
    """Model configuration for mlp-tts.
    """
    def __init__(self, mel: int):
        """Initializer.
        Args:
            mel: size of the output channels.
        """
        self.mel = mel

        self.embedding = 256
        self.prenet_kernels = 3

        self.channels = 512  # 768

        self.text_layers = 12
        self.text_ch_hiddens = self.channels * 4
        self.text_kernels = 192
        self.text_strides = self.text_kernels // 4
        self.text_tp_hiddens = 384
        self.text_dropout = 0.5  # 0.

        self.dur_channels = 128
        self.dur_kernels = [5, 3, 1]
        self.dur_dropout = 0.5  # 0.

        self.reg_conv = 8
        self.reg_kernels = 3
        self.reg_mlp = 16

        self.mel_layers = 12
        self.mel_ch_hiddens = self.channels * 4
        self.mel_kernels = 192
        self.mel_strides = self.text_kernels // 4
        self.mel_tp_hiddens = 384
        self.mel_dropout = 0.5  # 0.
