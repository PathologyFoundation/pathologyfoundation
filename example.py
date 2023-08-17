import pathologyfoundation as pf
plip = pf.model_zoo("PLIP-ViT-B-32")

# Example 1. Embed image(s)
import pandas as pd

"""
The image can be:
    - numpy array (W * H * 3)
    - numpy arrays (N * W * H * 3)
    - PIL Images (Image, or list of Images)
    - local/http URL (string)
    - local/http URLs (list of strings)
"""

example_image = "https://pbs.twimg.com/media/F3puqTTW8AEnB2F?format=jpg&name=large"
image_embeddings = plip.embed_images(example_image, normalize=True)


# Example 2. Embed text(s)
"""
The text can be:
    - String
    - list of Strings
"""
example_text = "An image of colorectal adenocarcinoma."
text_embeddings = plip.embed_texts(example_text, normalize=True)




# Example 3. Fine-tune PLIP as an image classifier
from models import finetuner
from PIL import Image
import torch
import os
df = pf.dataset.load_data.load_example(dataset_name="CRC-VAL-HE-7K")
df_image_label = df[["image", "label"]].sample(45)

print(df_image_label)

clf = finetuner.FineTuner(backbone=plip.model,
                           preprocess=plip.preprocess,
                           num_classes=len(df_image_label["label"].unique()),
                           freeze_vit=False,
                           random_state=0
                           )

clf.train(df_image_label,
          validation_split=0,
          batch_size=32,
          num_workers=1,
          lr=1e-5,
          nepochs=3
          )

# Save current fine-tuned model
clf.save_model(os.path.join(pf.utils.get_default_cache_dir(), 'model_statedict_epoch.pth'))

# Predict a single image
image = Image.open(df_image_label["image"].values[0])
proba = clf.predict(image)
predicted_class_name = df.loc[df['label'] == torch.argmax(proba).item(), 'class'][0]
print(f"Prediction: {predicted_class_name}")

# Generate image embedding
image = Image.open(df_image_label["image"].values[0])
image_embeddings = clf.extract_embedding(image)
