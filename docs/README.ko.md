# RVC Inference

[**English**](./README.md) | [**中文简体**](./docs/README.ch.md) | [**日本語**](./docs/README.ja.md) | [**한국어**](./docs/README.ko.md) | [**Français**](./docs/README.fr.md)| [**Türkçe**](./docs/README.tr.md)
------
GPT-4가 제공한 번역입니다.

## 설치
Python 3.11+를 사용하는 경우 아직 fairseq이 3.11과 호환되지 않으므로 먼저 fairseq 포크를 설치하세요.
```bash
pip install git+https://github.com/One-sixth/fairseq.git
```

아래와 같이 레포지토리를 pip 설치하면 모든 의존성이 자동으로 설치됩니다.
```bash
pip install git+https://github.com/CircuitCM/RVC-inference.git
```
기본적으로 pypi는 pytorch cpu 빌드를 설치합니다. Nvidia 또는 AMD를 사용하여 gpu용으로 설치하려면 https://pytorch.org/get-started/locally/를 방문하여 gpu용 `torch`와 `torchaudio`를 _이 라이브러리를 설치하기 전에_ pip로 설치하세요.

Python 3.8-3.12 버전은 지원되어야 하지만 테스트된 것은 3.11 뿐입니다. 설치나 호환성에 문제가 있으면 문제를 제기해 주시면 수정사항을 빠르게 반영하겠습니다.
수정 및 개선을 위한 PR은 환영합니다.

## 사용법
먼저 선택적 환경 변수를 설정하세요:
```python
import os
os.environ['RVC_MODELDIR']='path/to/rvc_model_dir' #모델.pth 파일이 저장된 곳.
os.environ['RVC_INDEXDIR']='path/to/rvc_index_dir' #모델.index 파일이 저장된 곳.
#오디오 출력 주파수, 기본값은 44100입니다.
os.environ['RVC_OUTPUTFREQ']='44100'
#출력 오디오 텐서가 완전히 로드될 때까지 차단해야 하는 경우 이를 무시할 수 있습니다. 하지만 더 큰 torch 파이프라인에서 실행하려면 False로 설정하면 성능이 약간 향상됩니다.
os.environ['RVC_RETURNBLOCKING']='True'
```
**환경 변수에 대한 참고사항:**
- `RVC_OUTPUTFREQ`와 `RVC_RETURNBLOCKING`은 `RVC` 클래스의 기본값으로 설정되지만 `self.outputfreq`와 `self.returnblocking`으로 인스턴스별로 재정의할 수 있습니다.
- `RVC_OUTPUTFREQ`를 `None`으로 설정하면 표준 리샘플링이 비활성화되고 모델의 기본 샘플 속도가 반환됩니다.
- `RVC_INDEXDIR`을 설정하지 않으면 `RVC` 클래스는 `RVC_MODELDIR`로, 마지막으로 모델 디렉토리의 절대 경로 `os.path.dirname(model_path)`로 대체됩니다.
- `RVC_MODELDIR`을 설정하지 않으면 arg `model`은 절대 경로여야 합니다.

모델 로드:
```python
from inferrvc import RVC
whis,obama=RVC('Whis.pth',index='added_IVF1972_Flat_nprobe_1_Whis_v2'),RVC(model='obama')

print(whis.name)
print('Paths',whis.model_path,whis.index_path)
print(obama.name)
print('Paths',obama.model_path,obama.index_path)
```
```text
모델: Whis, 인덱스: added_IVF1972_Flat_nprobe_1_Whis_v2
경로 Z:\Models\RVC\Models\Whis.pth Z:\Models\RVC\Indexes\added_IVF1972_Flat_nprobe_1_Whis_v2.index
모델: obama, 인덱스: obama
경로 Z:\Models\RVC\Models\obama.pth Z:\Models\RVC\Indexes\obama.index
```

인퍼런싱 실행:
```python
from inferrvc import load_torchaudio
aud,sr = load_torchaudio('path/to/audio.wav')

paudio1=whis(aud,f0_up

_key=6,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.75)
paudio2=obama(aud,5,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.9)

import soundfile as sf

sf.write('path/to/audio_whis.wav',paudio1,44100)
sf.write('path/to/audio_obama.wav',paudio2,44100)
```
[Whis 예제.](./docs/audio_whis.wav)  
[Obama 예제.](./docs/audio_obama.wav)

### 원래 레포지토리에서 변경된 사항:
 - 인퍼런싱과 관련 없는 대부분의 코드를 제거했습니다. 이제 의존성이 훨씬 적습니다.
 - 간소화된 인퍼런스 클래스와 파이프라인을 만들었습니다.
 - 성능 및 메모리 효율성 향상.
 - 일반 모델은 이제 `huggingface_hub`에 의해 관리되며 `HF_HOME` 환경 변수 경로를 통해 캐시됩니다.
 - RVC 모델 디렉토리와 파일에 대한 유연한 참조.
 - 버터워스 필터는 기본적으로 비활성화되었으며, 품질이 약간 떨어질 수 있으므로 일반적으로 차이가 없습니다. `inferrvc.pipeline.enable_butterfilter=True`로 활성화할 수 있습니다.

### 할 일:
- [ ] 다양한 파이썬 버전 테스트.
- [ ] 다양한 OS와 피치 추정기 테스트. (다른 추정기는 포팅되었지만 RMVPE만 테스트되었으며, 이것이 가장 좋습니다)
- [ ] 남은 작업을 단일 주요 장치(예: gpu)로 이동하여 메모리 전송으로 인한 지연과 속도 저하를 줄입니다.
  - [ ] 남은 numpy 코드를 `torch.where`와 `torch.masked_select`의 torch 등가물로 교체합니다.
  - [ ] gpu 장치용 pytorch로 인덱스 마스크를 다시 구현합니다.
- [ ] 가능하다면 torch 2.0 .compile()을 사용하여 v1/v2 모델의 속도를 높입니다.