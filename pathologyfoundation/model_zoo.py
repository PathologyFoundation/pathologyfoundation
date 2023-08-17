#from .models import PLIP_ViT
from models import PLIP_ViT
from .utils import get_default_cache_dir


def model_zoo(model_name, device=None, cache_dir=None):
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
        print(f"Use default cache dir: {cache_dir}")

    if model_name == "PLIP-ViT-B-32":
        return PLIP_ViT(model_name, device, cache_dir)
    else:
        raise ValueError(f"Model {model_name} not found in the model zoo.")
