# RVC Inference

[**English**](../README.md) | [**中文简体**](./README.ch.md) | [**日本語**](./README.ja.md) | [**한국어**](./README.ko.md) | [**Français**](./README.fr.md)| [**Türkçe**](./README.tr.md)
------
Traductions fournies par GPT-4.

## Installation
Si vous utilisez Python 3.11+, installez d'abord le fork fairseq car fairseq n'est pas encore compatible avec 3.11.
```bash
pip install https://github.com/One-sixth/fairseq/archive/main.zip
```

Installez le dépôt avec Pip comme ci-dessous et toutes les dépendances seront installées automatiquement.
```bash
pip install https://github.com/CircuitCM/RVC-inference/raw/main/dist/inferrvc-1.0-py3-none-any.whl
```
Par défaut, pypi installe la construction pytorch pour CPU. Pour installer pour GPU utilisant Nvidia ou AMD, visitez https://pytorch.org/get-started/locally/ et installez avec pip `torch` et `torchaudio` avec GPU _avant_ d'installer cette bibliothèque.

Le support devrait être disponible pour Python 3.8-3.12 mais seul 3.11 a été testé. Si vous rencontrez des problèmes avec l'installation ou la compatibilité, veuillez ouvrir un problème et je publierai des corrections.
Les PR avec des corrections et améliorations sont les bienvenues.

## Utilisation
Définissez d'abord les variables d'environnement optionnelles :
```python
import os
os.environ['RVC_MODELDIR']='chemin/vers/rvc_model_dir' #où les fichiers model.pth sont stockés.
os.environ['RVC_INDEXDIR']='chemin/vers/rvc_index_dir' #où les fichiers model.index sont stockés.
#la fréquence de sortie audio, par défaut est 44100.
os.environ['RVC_OUTPUTFREQ']='44100'
#Si le tenseur audio de sortie doit bloquer jusqu'à être complètement chargé, cela peut être ignoré. Mais si vous voulez l'exécuter dans un pipeline torch plus grand, le régler sur False améliorera un peu les performances.
os.environ['RVC_RETURNBLOCKING']='True'
```
**Notes sur les variables d'environnement :**
- Les `RVC_OUTPUTFREQ` et `RVC_RETURNBLOCKING` définissent les valeurs par défaut pour la classe `RVC`, mais elles peuvent être outrepassées par instance avec `self.outputfreq` et `self.returnblocking`.
- Régler `RVC_OUTPUTFREQ` sur `None` désactivera le rééchantillonnage standard et renverra la fréquence d'échantillonnage native du modèle.
- Si vous ne définissez pas `RVC_INDEXDIR`, la classe `RVC` se rabattra sur `RVC_MODELDIR` et enfin sur le chemin absolu du répertoire du modèle `os.path.dirname(model_path)`.
- Si vous ne définissez pas `RVC_MODELDIR`, alors l'argument `model` doit être un chemin absolu.

Chargement des modèles :
```python
from inferrvc import RVC
whis,obama=RVC('Whis.pth',index='added_IVF1972_Flat_nprobe_1_Whis_v2'),RVC(model='obama')

print(whis.name)
print('Chemins',whis.model_path,whis.index_path)
print(obama.name)
print('Chemins',obama.model_path,obama.index_path)
```
```text
Modèle : Whis, Index : added_IVF1972_Flat_nprobe_1_Whis_v2
Chemins Z:\Models\RVC\Models\Whis.pth Z:\Models\RVC\Indexes\added_IVF1972_Flat_nprobe_1_Whis_v2.index
Modèle : obama, Index : obama
Chemins Z:\Models\RVC\Models\obama.pth Z:\Models\RVC\Indexes\obama.index
```

Exécuter l'inférence :
```python
from inferrvc import load_torchaudio
aud,sr = load_torchaudio('chemin/vers/audio.wav')

paudio1=whis(aud,f0_up_key=6,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.75)
paudio2=obama(aud,5,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.9)

import soundfile as sf

sf.write('chemin/vers/audio_whis.wav',paudio1,44100)
sf.write('chemin/vers/audio_obama.wav',paudio2,44100)
```
[Exemple Whis.](./docs/audio_whis.wav)  
[Exemple Obama.](./docs/audio_obama.wav)

### Chang

ements par rapport au dépôt original :
 - Suppression de la plupart du code non lié à l'inférence. Maintenant beaucoup moins de dépendances.
 - Création d'une classe et d'un pipeline d'inférence simplifiés.
 - Améliorations de la performance et de l'efficacité de la mémoire.
 - Les modèles génériques sont maintenant gérés par `huggingface_hub` et mis en cache via la variable d'environnement `HF_HOME`.
 - Référencement flexible du répertoire et des fichiers du modèle RVC.
 - Désactivation du filtre butterworth par défaut car il n'y a généralement pas de différence et cela pourrait légèrement réduire la qualité. Peut être activé avec `inferrvc.pipeline.enable_butterfilter=True`.

### Tâches à faire :
- [ ] Tester différentes versions de Python.
- [ ] Tester différents systèmes d'exploitation et estimateurs de hauteur. (Les autres estimateurs devraient être portés mais seul RMVPE a été testé, c'est le meilleur)
- [ ] Déplacer les opérations restantes sur le dispositif principal unique (par exemple GPU), pour réduire la latence et le ralentissement résultant des transferts de mémoire.
  - [ ] Remplacer le code numpy restant par des équivalents torch `torch.where` et `torch.masked_select`.
  - [ ] Réimplémenter le masque d'index avec pytorch pour les dispositifs GPU.
- [ ] Utiliser la méthode .compile() de torch 2.0 pour accélérer les modèles v1/v2 si possible.