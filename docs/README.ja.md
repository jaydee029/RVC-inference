
# RVC Inference
[**English**](./README.md) | [**中文简体**](./docs/README.ch.md) | [**日本語**](./docs/README.ja.md) | [**한국어**](./docs/README.ko.md) | [**Français**](./docs/README.fr.md)| [**Türkçe**](./docs/README.tr.md)
------
翻訳提供：GPT-4

## インストール
Python 3.11+ を使用する場合は、fairseq がまだ 3.11 と互換性がないため、まず fairseq のフォークをインストールしてください。
```bash
pip install git+https://github.com/One-sixth/fairseq.git
```

以下のようにリポジトリを Pip インストールすると、必要な依存関係が自動的にインストールされます。
```bash
pip install git+https://github.com/CircuitCM/RVC-inference.git
```
デフォルトでは pypi は pytorch の CPU ビルドをインストールします。Nvidia または AMD を使用して GPU 用にインストールするには、https://pytorch.org/get-started/locally/ を訪れて、このライブラリをインストールする _前に_ `torch` と `torchaudio` を GPU 付きで pip インストールしてください。

Python 3.8-3.12 でサポートされるべきですが、テストされたのは 3.11 のみです。インストールや互換性に問題がある場合は、問題を報告してください。修正をプッシュします。
修正や改善の PR も歓迎します。

## 使い方
最初にオプションの環境変数を設定します：
```python
import os
os.environ['RVC_MODELDIR']='path/to/rvc_model_dir' #モデルの .pth ファイルが保存されている場所。
os.environ['RVC_INDEXDIR']='path/to/rvc_index_dir' #モデルの .index ファイルが保存されている場所。
#オーディオ出力周波数、デフォルトは 44100。
os.environ['RVC_OUTPUTFREQ']='44100'
#出力オーディオテンソルが完全にロードされるまでブロックする必要がある場合、これは無視して構いません。しかし、より大きな torch パイプラインで実行したい場合は、False に設定するとパフォーマンスが少し向上します。
os.environ['RVC_RETURNBLOCKING']='True'
```
**環境変数に関する注意事項：**
- `RVC_OUTPUTFREQ` と `RVC_RETURNBLOCKING` は `RVC` クラスのデフォルトを設定しますが、`self.outputfreq` と `self.returnblocking` でインスタンスごとに上書きすることができます。
- `RVC_OUTPUTFREQ` を `None` に設定すると、標準のリサンプリングが無効になり、モデルのネイティブサンプルレートが返されます。
- `RVC_INDEXDIR` を設定しない場合、`RVC` クラスは `RVC_MODELDIR` にフォールバックし、最終的にはモデルディレクトリの絶対パス `os.path.dirname(model_path)` にフォールバックします。
- `RVC_MODELDIR` を設定しない場合、引数 `model` は絶対パスでなければなりません。

モデルのロード：
```python
from inferrvc import RVC
whis,obama=RVC('Whis.pth',index='added_IVF1972_Flat_nprobe_1_Whis_v2'),RVC(model='obama')

print(whis.name)
print('Paths',whis.model_path,whis.index_path)
print(obama.name)
print('Paths',obama.model_path,obama.index_path)
```
```text
Model: Whis, Index: added_IVF1972_Flat_nprobe_1_Whis_v2
Paths Z:\Models\RVC\Models\Whis.pth Z:\Models\RVC\Indexes\added_IVF1972_F

lat_nprobe_1_Whis_v2.index
Model: obama, Index: obama
Paths Z:\Models\RVC\Models\obama.pth Z:\Models\RVC\Indexes\obama.index
```

推論の実行：
```python
from inferrvc import load_torchaudio
aud,sr = load_torchaudio('path/to/audio.wav')

paudio1=whis(aud,f0_up_key=6,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.75)
paudio2=obama(aud,5,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.9)

import soundfile as sf

sf.write('path/to/audio_whis.wav',paudio1,44100)
sf.write('path/to/audio_obama.wav',paudio2,44100)
```
[Whis の例。](./docs/audio_whis.wav)  
[Obama の例。](./docs/audio_obama.wav)

### 元のリポジトリからの変更点：
 - 推論に関連しないほとんどのコードを削除しました。これで依存関係が大幅に減ります。
 - ストリームライン化された推論クラスとパイプラインを作成しました。
 - パフォーマンスとメモリ効率の向上。
 - 汎用モデルは現在 `huggingface_hub` によって管理され、`HF_HOME` 環境変数を通じてキャッシュされます。
 - RVC モデルディレクトリとファイルへの柔軟な参照。
 - バターワースフィルターは通常差がなく、若干の品質低下を引き起こす可能性があるため、デフォルトで無効になっています。`inferrvc.pipeline.enable_butterfilter=True` で有効にできます。

### Todo：
- [ ] 異なる Python バージョンをテストする。
- [ ] 異なる OS とピッチ推定器をテストする。（他の推定器は移植されるべきですが、RMVPE のみがテストされました。これは最高です）
- [ ] 残りの操作を単一のプライマリデバイス（例：GPU）に移動し、メモリ転送による遅延とスローダウンを減らします。
  - [ ] 残りの numpy コードを `torch.where` と `torch.masked_select` の torch 同等品に置き換える。
  - [ ] インデックスマスクを GPU デバイス用の pytorch で再実装する。
- [ ] torch 2.0 の .compile() を使用して、可能であれば v1/v2 モデルを高速化する。