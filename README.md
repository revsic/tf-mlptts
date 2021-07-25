# tf-mlptts

(PENDING) Tensorflow implementation of MLP-Mixer based TTS.

It is just for testing mlp-based architecture and TTS compatibility. \
It generates mel-spectrogram naively, which cannot be recovered to audio by vocoder. \
I leave some dynamic length MLP-like modules on [mlp.py](./mlptts/mlpmixer/mlp.py), and wish to complete this project in a short days.

Experimental detail is described on [Philosophy](#philosophy).

## Requirements

Tested in python 3.8.5 windows10 conda environment, [requirements.txt](./requirements.txt)

## Usage

To download LJ-Speech dataset, run under script.

Dataset will be downloaded in '~/tensorflow_datasets' in tfrecord format. If you want to change the download directory, specify `data_dir` parameter of `LJSpeech` initializer.

```python
from speechset.datasets import LJSpeech

# lj = LJSpeech(data_dir=path, download=True)
lj = LJSpeech(download=True) 
```

To train model, run [train.py](./train.py). 

Checkpoint will be written on `TrainConfig.ckpt`, tensorboard summary on `TrainConfig.log`.

```bash
python train.py
tensorboard --logdir .\log
```

If you want to train model from raw audio, specify audio directory and turn on the flag `--from-raw`.

```bash
python .\train.py --data-dir D:\LJSpeech-1.1\wavs --from-raw
```

To start to train from previous checkpoint, `--load-epoch` is available.

```bash
python .\train.py --load-epoch 20 --config D:\tf\ckpt\mlptts.json
```

Pretrained checkpoints are relased on [releases](https://github.com/revsic/tf-mlptts/releases).

To use pretrained model, download files and unzip it. Followings are sample script.

```py
from config import Config
from mlptts import MLPTextToSpeech

with open('mlptts.json') as f:
    config = Config.load(json.load(f))

tts = MLPTextToSpeech(config.model)
tts.restore('./mlptts_299.ckpt-1').expect_partial()
```

## Learning Curve

train LJSpeech 300 epochs with [tf-diffwave](https://github.com/revsic/tf-diffwave)

![loss](./rsrc/loss.png)
![align](./rsrc/align.png)
![mel](./rsrc/mel.png)


## Philosophy

So far, there is a lot of TTS researches, especially parallel feed-forward models are trends these days. MLP-TTS is one of them, parallel feed-forward models, with end-to-end duration modeling introduced by Parallel Tacotron[3].

On the other side, MLP-Mixer[1] and ResMLP[2] introduce MLP-only architectures for NN backbones.

Since transformer and convolutions are major backbones on current TTS fields, I try to research the other backbones which can be compatible with TTS. And MLP-TTS is experiments for TTS with MLP-based architectures.

### 1. Dynamic length MLP

The first issue was how to model MLP for dynamic length inputs.

MLP-Mixer[1] and ResMLP[2] assume a fixed number of patches and it is possible to use MLP on the tempooral axis. But in the TTS case, it accepts dynamic size inputs and temporal-level MLP is impossible.

I propose some dynamic-length MLP-like architectures, which can be found on [mlp.py](./mlptts/mlpmixer/mlp.py).

A. ConvMLP

`ConvMLP` uses wide lnegth MLP on temporal axis. To deal with temporal dynamicity, it makes fixed length frames with overlapped regions, which can be simply found on STFT (Short-term Fourier Transform). After two MLP layers, it operates overlap-and-add to make the single sequence. 

For computational efficiency, 2D convolution and transposed convolution are chosen, but it is not practical since large size of memory is required (batch x num_frames x frame_size). And since we choose transposed convolution for cuda-accelaration of overlap-and-add, window function cannot be adaptable in operation sequence (ex. hann window).

B. TemporalConv

It is simple convolution only acts on temporal axis. To prevent the operation on channel axis, Conv2D with [S, 1] stride and [K, 1] kernel is used on expanded input features [B, T, C, 1].

C. DynTemporalMLP

It computes dynamic MLP weights from input features. For simplicity, mlp operates on transposed and concatenated features.
> [B, T, C, 1] x [B, T, 1, C] = [B, T, T, Cx2] -> [B, T, T, 1]

It has two philosophical differences between MLP-Mixer.

First, Mixer separates the module into two exclusive operations, channel-level MLP and temporal-level MLP. But `DynTemporalMLP` acts on the temporal axis for computing dynamic weights. 

Second, since MLP learns the weights from individual positions, additional positional information is unnecessary. But `DynTemporalMLP` requires positional embeddings since it just computes the weights from two features of different positions, permutation invariant.

---
In baseline, mlp-tts uses large size `TemporalConv` for temporal transformation.

### 2. MLP-Mixer vs ResMLP

In experiments, there are too many dense operations and training procedure was instable. I met many NaNs on train.

I check some possibilities to stabilize training procedure.

A. Layer normalization, MLP-Mixer

Like Transformer (Vaswani et al., 2017), MLP-Mixer[1] uses layer normalization on each residual blocks. In mlptts, it cannot stabilize the training.

B. Affine-transform, ResMLP

In paper of CaiT (Touvron et al., 2021), it introduces LayerScale, which use affine transform with small value of scale factor. ResMLP[2] uses normalization-free LayerScale and mlptts can be stabilized.

C. DDI-ActNorm, RescaleNet

RescaleNet[5] introduces DDI(data-dependent initialization) based debiasing for the solution of dead ReLU. Inspired by RescaleNet, I try to alter the affine transform with DDI-based activation normalization (Glow, Kingma et al., 2018). But effect was minimal and the key was the small value of scaling factor.

---
In baseline, mlp-tts uses affine transform with the small value of scaling factor.

### 3. Gaussian upsampling vs Regulator

From FastSpeech (Ren et al., 2019), explicit duration modeler is common on TTS in nowadays. MLP-TTS also uses an explicit duration modeler and train end-to-end with a gaussian upsampler.

Regulator from Parallel Tacotron 2[4] is a more recent contribution, but it is unstable on mlp-tts. I think regulator was possible on parallel tacotron 2 since it assumes the locality with light-weight dynamic convolution, so that monotonicity of alignment can be inferenced from encoded features. However, MLP-TTS uses MLP and large-size temporal convolution, so that locality assumption can be freed and monotonicity cannot be inferenced from features (only on my thought).

So I explicitly assume the monotonicity with the gaussian upsampling mechanism (Parallel Tacotron[3]), and it can stabilize the mlp-tts training.

### 4. Future works

It cannot recover mel-spectrogram in detail. There are so many factors that can lead to this failure, temporal convolution, gaussian upsampler, and hyperparameters, etc. Now this repository is pended due to personal circumstances. I wish to complete this project in a short day.

### Reference
[1] MLP-Mixer: An all-MLP Architecture for Vision, Tolstikhin et al., 2021. \
[2] ResMLP: Feedforward networks for image classification with data-efficient training, Touvron et al., 2021. \
[3] Parallel Tacotron: Non-Autoregressive and Controllable TTS, Elias et al., 2020. \
[4] Parallel Tacotron 2: A Non-Autoregressive Neural TTS Model with Differentiable Duration Modeling, Elias et al., 2021. \
[5] Is normalization indispensable for training deep neural networks, Shao et al., 2020.
