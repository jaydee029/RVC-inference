# RVC Inference

[**English**](./README.md) | [**中文简体**](./docs/README.ch.md) | [**日本語**](./docs/README.ja.md) | [**한국어**](./docs/README.ko.md) | [**Français**](./docs/README.fr.md)| [**Türkçe**](./docs/README.tr.md)
------
GPT-4 tarafından sağlanan çeviriler.

## Kurulum
Python 3.11+ kullanıyorsanız, önce fairseq şubesini yükleyin, çünkü fairseq henüz 3.11 ile uyumlu değil.
```bash
pip install git+https://github.com/One-sixth/fairseq.git
```

Aşağıdaki gibi repoyu pip ile yükleyin ve tüm bağımlılıklar otomatik olarak yüklenecektir.
```bash
pip install git+https://github.com/CircuitCM/RVC-inference.git
```
Varsayılan olarak pypi, pytorch'un cpu sürümünü yükler. Nvidia veya AMD kullanarak gpu için yüklemek için, https://pytorch.org/get-started/locally/ adresini ziyaret edin ve bu kütüphaneyi yüklemekten _önce_ `torch` ve `torchaudio`'yu gpu ile pip yükleyin.

Destek, Python 3.8-3.12 için mevcut olmalıdır, ancak sadece 3.11 test edildi. Kurulum veya uyumlulukla ilgili herhangi bir sorun olursa lütfen bir sorun açın ve düzeltmeleri yayınlayacağım.
Düzeltmeler ve iyileştirmeler içeren PR'lar hoş karşılanır.

## Kullanım
Önce isteğe bağlı ortam değişkenlerini ayarlayın:
```python
import os
os.environ['RVC_MODELDIR']='path/to/rvc_model_dir' #model.pth dosyalarının saklandığı yer.
os.environ['RVC_INDEXDIR']='path/to/rvc_index_dir' #model.index dosyalarının saklandığı yer.
#ses çıkış frekansı, varsayılan 44100.
os.environ['RVC_OUTPUTFREQ']='44100'
#Çıktı ses tensörü tamamen yüklenene kadar bloklanmalı mı, bu görmezden gelinebilir. Ancak daha büyük bir torch hattında çalıştırmak istiyorsanız, False olarak ayarlamak performansı biraz iyileştirecektir.
os.environ['RVC_RETURNBLOCKING']='True'
```
**Ortam değişkenleriyle ilgili notlar:**
- Hem `RVC_OUTPUTFREQ` hem de `RVC_RETURNBLOCKING`, `RVC` sınıfı için varsayılanları belirler, ancak `self.outputfreq` ve `self.returnblocking` ile örnek başına geçersiz kılınabilirler.
- `RVC_OUTPUTFREQ`'yi `None` olarak ayarlamak, standart yeniden örnekleme işlemi yapmayacak ve modelin yerel örneklem oranını döndürecektir.
- `RVC_INDEXDIR` ayarlamazsanız, `RVC` sınıfı `RVC_MODELDIR`'a ve son olarak model dizininin mutlak yoluna `os.path.dirname(model_path)` geri dönecektir.
- `RVC_MODELDIR` ayarlamazsanız, argüman `model` mutlak bir yol olmalıdır.

Modelleri yükleyin:
```python
from inferrvc import RVC
whis,obama=RVC('Whis.pth',index='added_IVF1972_Flat_nprobe_1_Whis_v2'),RVC(model='obama')

print(whis.name)
print('Yollar',whis.model_path,whis.index_path)
print(obama.name)
print('Yollar',obama.model_path,obama.index_path)
```
```text
Model: Whis, İndeks: added_IVF1972_Flat_nprobe_1_Whis_v2
Yollar Z:\Models\RVC\Models\Whis.pth Z:\Models\RVC\Indexes\added_IVF1972_Flat_nprobe_1_Whis_v2.index
Model: obama, İndeks: obama
Yollar Z:\Models\RVC\Models\obama.pth Z:\Models\RVC\Indexes\obama.index
```

Çıkarım Yapma:
```python
from inferrvc import load_torchaudio
aud,sr = load_torchaudio('path/to/audio.wav')

paudio1=whis(aud,f0_up_key=6,output_device='cpu',output_volume=RVC

.MATCH_ORIGINAL,index_rate=.75)
paudio2=obama(aud,5,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.9)

import soundfile as sf

sf.write('path/to/audio_whis.wav',paudio1,44100)
sf.write('path/to/audio_obama.wav',paudio2,44100)
```
[Whis örneği.](./docs/audio_whis.wav)  
[Obama örneği.](./docs/audio_obama.wav)

### Orijinal repodaki değişiklikler:
 - Çıkarım ile ilgili olmayan çoğu kodu kaldırdı. Artık çok daha az bağımlılık var.
 - Akıcı bir çıkarım sınıfı ve iş akışı oluşturuldu.
 - Performans ve bellek verimliliği iyileştirmeleri.
 - Genel modeller artık `huggingface_hub` tarafından yönetiliyor ve `HF_HOME` ortam değişkeni yoluyla önbelleğe alınıyor.
 - RVC model dizini ve dosyalarına esnek referans.
 - Butterworth filtresi genellikle fark yaratmadığı ve kaliteyi hafifçe düşürebileceği için varsayılan olarak devre dışı bırakıldı. `inferrvc.pipeline.enable_butterfilter=True` ile etkinleştirilebilir.

### Yapılacaklar:
- [ ] Farklı python sürümlerini test etmek.
- [ ] Farklı işletim sistemlerini ve perde tahmincilerini test etmek. (Diğer tahminciler taşınmış olmalı ancak sadece RMVPE test edildi, bu en iyisidir)
- [ ] Kalan işlemleri tek bir ana cihaza (ör. gpu) taşımak, bellek transferlerinden kaynaklanan gecikmeyi ve yavaşlamayı azaltmak.
  - [ ] Kalan numpy kodlarını `torch.where` ve `torch.masked_select` torch eşdeğerleri ile değiştirmek.
  - [ ] İndeks maskesini gpu cihazlar için pytorch ile yeniden uygulamak.
- [ ] V1/V2 modellerini hızlandırmak için mümkünse torch 2.0 .compile() kullanmak.