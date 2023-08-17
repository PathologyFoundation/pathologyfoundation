import appdirs
#from .models import PLIP_ViT
from models import PLIP_ViT

app_name = "PathologyFoundation"
default_cache_dir = appdirs.user_cache_dir(app_name)

def model_zoo(model_name, device=None, cache_dir=None):
    if cache_dir is None:
        print(f"Use default cache dir: {default_cache_dir}")
        cache_dir = default_cache_dir

    if model_name == "PLIP-ViT-B-32":
        return PLIP_ViT(model_name, device, cache_dir)
    else:
        raise ValueError(f"Model {model_name} not found in the model zoo.")
