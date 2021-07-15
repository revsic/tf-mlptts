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
        self.dropout = 0.

        self.text_layers = 8  # 12
        self.text_hiddens = self.channels  # x 4
        self.text_kernels = 11

        self.res_channels = self.mel
        self.latent_channels = 16

        self.res_layers = 8
        self.res_hiddens = self.mel
        self.res_kernels = 11

        self.dur_layers = 3
        self.dur_hiddens = self.channels
        self.dur_kernels = 3

        self.reg_conv = 8
        self.reg_kernels = 3
        self.reg_mlp = 16
        self.reg_aux = 2

        self.mel_layers = 8
        self.mel_hiddens = self.channels  # desired 4
        self.mel_kernels = 11
