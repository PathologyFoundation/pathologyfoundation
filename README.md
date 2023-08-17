# PathologyFoundation

<div style="text-align:center"><img src="figures/logo.png" width=300/></div>

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/pathologyfoundation/PathologyFoundation/blob/master/LICENSE)

## Installation

* From PyPI

```bash
pip install pathologyfoundation
```

## Examples

### Example 1. Load plip model and embed image(s)

```python
import pathologyfoundation as pf
plip = pf.model_zoo("PLIP-ViT-B-32")
example_image = "https://pbs.twimg.com/media/F3puqTTW8AEnB2F?format=jpg&name=large"
image_embeddings = plip.embed_images(example_image, normalize=True)
```

The image format can be:
- numpy array (W * H * 3)
- numpy arrays (N * W * H * 3)
- PIL Images (Image, or list of Images)
- local/http URL (string)
- local/http URLs (list of strings)


### Example 2. Load plip model and embed text(s)

```python
import pathologyfoundation as pf
plip = pf.model_zoo("PLIP-ViT-B-32")
example_text = "An image of colorectal adenocarcinoma."
text_embeddings = plip.embed_texts(example_text, normalize=True)
```

The text format can be:
- String
- list of Strings




### Example 3. Fine-tune PLIP as an image classifier

Step 1. Download example data `CRC-VAL-HE-7K`.

```python
import pathologyfoundation as pf
from PIL import Image
import torch
import os

plip = pf.model_zoo("PLIP-ViT-B-32")
df = pf.dataset.load_data.load_example(dataset_name="CRC-VAL-HE-7K")
df_image_label = df[["image", "label"]].sample(45) # Subsample few images in this tutorial.
print(df_image_label)
```

Step 2. Initialize a classifier with PLIP model as backbone.

```python
clf = pf.models.finetuner.FineTuner(backbone=plip.model,
                           preprocess=plip.preprocess,
                           num_classes=len(df_image_label["label"].unique()),
                           freeze_vit=False,
                           random_state=0
                           )
```

Step 3. Start training the model.

```python
clf.train(df_image_label,
          validation_split=0.1,
          batch_size=32,
          num_workers=1,
          lr=1e-5,
          nepochs=10
          )
```

Optional 1: Save current fine-tuned model.

```python
clf.save_model(os.path.join(pf.utils.get_default_cache_dir(), 'model_statedict_epoch.pth'))
```


Optional 2: Predict a single image.

```python
image = Image.open(df_image_label["image"].values[0])
proba = clf.predict(image)
predicted_class_name = df.loc[df['label'] == torch.argmax(proba).item(), 'class'].values[0]
print(f"Prediction: {predicted_class_name}")
```

Optional 3: Generate image embedding.

```python
image = Image.open(df_image_label["image"].values[0])
image_embeddings = clf.extract_embedding(image)
```
