"""High performance RVC inferencing, intended for multiple instances in memory at once. Also includes the latest pitch estimator RMVPE, Python 3.8-3.11 compatible, pip installable, memory + performance improvements in the pipeline and model usage."""

__version__ = '1.0'

import dotenv as _dotenv
_dotenv.load_dotenv(override=False)

from .modules import RVC,ResampleCache,download_models,load_torchaudio