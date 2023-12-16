# RVC Inference
[**English**](../README.md) | [**中文简体**](./README.ch.md) | [**日本語**](./README.ja.md) | [**한국어**](./README.ko.md) | [**Français**](./README.fr.md)| [**Türkçe**](./README.tr.md)
------
翻译由 GPT-4 提供。
## 安装
如果使用 Python 3.11 及以上版本，请首先安装 fairseq 分支，因为 fairseq 尚未兼容 3.11 版本。
```bash
pip install git+https://github.com/One-sixth/fairseq.git
```

使用下面的命令安装本仓库，所有依赖项将自动安装。
```bash
pip install git+https://github.com/CircuitCM/RVC-inference.git
```
默认情况下，pypi 会安装 PyTorch 的 CPU 版本。如果要在 Nvidia 或 AMD 的 GPU 上安装，请访问 https://pytorch.org/get-started/locally/，然后在安装此库之前使用 pip 安装 `torch` 和 `torchaudio` 的 GPU 版本。

此库应该支持 Python 3.8 至 3.12 版本，但只测试了 3.11 版本。如果安装或兼容性有任何问题，请开设一个 issue，我将推出修复版本。
欢迎提交带有修复和改进的 PR（Pull Request）。

## 使用方法
首先设置可选的环境变量：
```python
import os
os.environ['RVC_MODELDIR']='path/to/rvc_model_dir' # 存储 model.pth 文件的位置。
os.environ['RVC_INDEXDIR']='path/to/rvc_index_dir' # 存储 model.index 文件的位置。
# 音频输出频率，默认为 44100。
os.environ['RVC_OUTPUTFREQ']='44100'
# 如果希望输出的音频张量完全加载后再进行阻塞，可以忽略此设置。但如果你想在更大的 torch 管道中运行，设置为 False 可略微提高性能。
os.environ['RVC_RETURNBLOCKING']='True'
```
**环境变量说明：**
- `RVC_OUTPUTFREQ` 和 `RVC_RETURNBLOCKING` 会为 `RVC` 类设置默认值，但可以在每个实例中用 `self.outputfreq` 和 `self.returnblocking` 进行覆盖。
- 将 `RVC_OUTPUTFREQ` 设置为 `None` 将禁用标准重采样，并返回模型的原生采样率。
- 如果你没有设置 `RVC_INDEXDIR`，`RVC` 类将回退到 `RVC_MODELDIR`，最后是模型目录的绝对路径 `os.path.dirname(model_path)`。
- 如果你没有设置 `RVC_MODELDIR`，那么参数 `model` 必须是一个绝对路径。

加载模型：
```python
from inferrvc import RVC
whis,obama=RVC('Whis.pth',index='added_IVF1972_Flat_nprobe_1_Whis_v2'),RVC(model='obama')

print(whis.name)
print('路径',whis.model_path,whis.index_path)
print(obama.name)
print('路径',obama.model_path,obama.index_path)
```
```text
模型：Whis，索引：added_IVF1972_Flat_nprobe_1_Whis_v2
路径 Z:\Models\RVC\Models\Whis.pth Z:\Models\RVC\Indexes\added_IVF1972_Flat_nprobe_1_Whis_v2.index
模型：obama，索引：obama
路径 Z:\Models\RVC\Models\obama.pth Z:\Models\RVC\Indexes\obama.index
```

执行推理：
```python
from inferrvc import load_torchaudio
aud,sr = load_torchaudio('path/to/audio.wav')

paudio1=whis(aud,f0_up_key=6,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.75)
paudio2=obama(aud,5,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.9)

import soundfile as sf

sf.write('path/to/audio_whis.wav',paudio1,44100)
sf.write('path/to/audio_obama.wav',paudio2,44100)
```
[Whis 示例。](./docs/audio_whis.wav)  
[Obama 示例。](./docs/audio_obama.wav)

### 与原始仓库的更改：
 - 移除了大部分与推理无关的代码。现在依赖项更少。
 - 创建

了一个简化的推理类和流程。
 - 性能和内存效率改进。
 - 通用模型现在由 `huggingface_hub` 管理，并通过 `HF_HOME` 环境变量进行缓存。
 - 灵活引用 RVC 模型目录和文件。
 - 默认禁用了巴特沃斯滤波器，因为通常没有区别，可能会略微降低质量。可以通过设置 `inferrvc.pipeline.enable_butterfilter=True` 来启用。

### 待办事项：
- [ ] 测试不同的 Python 版本。
- [ ] 测试不同的操作系统和音高估计器。（其他估计器应该已被移植，但只测试了 RMVPE，它是最佳的）
- [ ] 将剩余操作移到单一主设备（例如 GPU），以减少延迟和因内存传输导致的减速。
  - [ ] 用 torch 的等价物 `torch.where` 和 `torch.masked_select` 替换剩余的 numpy 代码。
  - [ ] 使用 pytorch 为 GPU 设备重新实现索引掩码。
- [ ] 如果可能，利用 torch 2.0 的 .compile() 加速 v1/v2 模型。