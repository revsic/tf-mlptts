class Config:
    """Model configuration for mlp-tts.
    """
    def __init__(self, vocabs: int, mel: int):
        """Initializer.
        Args:
            vocabs: size of the vocabularies.
            mel: size of the output channels.
        """
        self.vocabs = vocabs
        self.mel = mel

        self.channels = 256  # 768
        self.eps = 1e-3

        self.text_layers = 6  # 12
        self.text_hiddens = self.channels  # x 4
        self.text_dropout = 0.

        self.res_channels = 16

        self.dur_layers = 2
        self.dur_hiddens = self.channels
        self.dur_dropout = 0.3

        self.reg_conv = 8
        self.reg_kernels = 3
        self.reg_mlp = 16
        self.reg_aux = 2

        self.mel_layers = 6
        self.mel_hiddens = self.channels  # desired 4
        self.mel_dropout = 0.
