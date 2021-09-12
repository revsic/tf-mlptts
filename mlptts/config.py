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

        self.text_layers = 8  # 12
        self.text_hiddens = 384  # Cx4
        self.text_kernels = 193

        self.res_channels = self.mel
        self.latent_channels = 16

        self.res_layers = 8
        self.res_hiddens = 128  # assume mel = 80
        self.res_kernels = 193

        self.ctc_layers = 2
        self.ctc_hiddens = 384
        self.ctc_factor = 2
        self.ctc_lambda = 1e-2

        self.dur_layers = 3
        self.dur_hiddens = self.channels
        self.dur_kernels = 3

        self.mel_layers = 12
        self.mel_hiddens = 384
        self.mel_kernels = 193
