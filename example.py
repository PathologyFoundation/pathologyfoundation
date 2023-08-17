#import pathologyfoundation as pf
#plip = pf.model_zoo("PLIP-ViT-B-32")
from model_zoo import model_zoo
plip = model_zoo("PLIP-ViT-B-32")

# Example 1. Embed image(s)
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