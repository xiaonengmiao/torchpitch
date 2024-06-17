# torchpitch
Fundamental frequency (F0) estimation using the YIN algorithm.

This package "translates" the [librosa.yin](https://github.com/librosa/librosa/blob/main/librosa/core/pitch.py) in torch.Tensor.

# usage

```python
import torchaudio
from torchpitch import yin
wave, sr = torchaudio.load("./data/data_aishell3/test/wav/SSB1831/SSB18310007.wav")

f0 = yin(wave.squeeze(0), fmin=20, fmax=2000, sr=sr)
```
