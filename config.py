import tensorflow as tf

from speechset import Config as DataConfig, AcousticDataset
from mlptts.config import Config as ModelConfig
from utils.warmup import Warmup


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self):
        self.hash = 'undefined'

        # optimizer
        self.lr_policy = 'fixed'
        self.learning_rate = 1e-3
        # self.lr_policy = 'warmup'
        # self.lr_params = {
        #     'learning_rate': 1e-4,
        #     'warmup_steps': 200,
        #     'alpha': 6.,
        # }

        self.beta1 = 0.9
        self.beta2 = 0.98
        self.eps = 1e-9

        # 13000:100
        self.split = 13000
        self.bufsiz = 48

        self.epoch = 200

        # path config
        self.log = './log'
        self.ckpt = '/hdd1/revsic/ckpt/mlptts'

        # model name
        self.name = 'mlptts-ab'

    def lr(self):
        """Generate proper learning rate scheduler.
        """
        mapper = {
            'expdecay': tf.keras.optimizers.schedules.ExponentialDecay,
            'piecewise': tf.keras.optimizers.schedules.PiecewiseConstantDecay,
            'warmup': Warmup,
        }
        if self.lr_policy == 'fixed':
            return self.learning_rate
        if self.lr_policy in mapper:
            return mapper[self.lr_policy](**self.lr_params)
        raise ValueError('invalid lr_policy')

class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig(AcousticDataset.VOCABS, self.data.mel)
        self.train = TrainConfig()

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
