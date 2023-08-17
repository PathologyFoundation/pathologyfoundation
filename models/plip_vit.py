import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image, UnidentifiedImageError
import numpy as np
import requests
from io import BytesIO
import os
from pathlib import Path
from hashlib import md5
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import clip

class ImageArrayDataset(Dataset):
    def __init__(self, list_of_images):
        self.images = list_of_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        return images

class CaptioningDataset(Dataset):
    def __init__(self, captions):
        self.caption = captions

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        caption = self.caption[idx]
        return caption



class PLIP_ViT:
    def __init__(self, model_name, device=None, cache_dir=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained("vinid/plip", cache_dir=cache_dir)
        def preprocess_wrapper(img):
            result = processor.image_processor(img)['pixel_values']
            return result
        self.preprocess = preprocess_wrapper

        self.model = self._load_model(cache_dir)
        

    def _load_model(self, cache_dir):
        if self.model_name == "PLIP-ViT-B-32":
            model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip", cache_dir=cache_dir)
            return model.to(self.device)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
            

    def preprocess_single_image(self, img):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            # Convert local or http URL to Image
            if isinstance(img, str):
                if os.path.exists(img):  # Local URL
                    img = Image.open(img)
                elif img.startswith('http'):  # Remote URL
                    response = requests.get(img, headers=headers)
                    img = Image.open(BytesIO(response.content))
                else:
                    raise ValueError("Invalid URL or path")

            # Convert numpy array to PIL Image
            elif isinstance(img, np.ndarray) and img.ndim == 3:
                img = Image.fromarray(img)

            ## Check if the image is corrupted
            #img.verify()

            # Now, preprocess the PIL Image
            result = self.preprocess(img)
            return result

        except (requests.RequestException, ValueError, UnidentifiedImageError, FileNotFoundError) as e:
            raise e

    def preprocess_images(self, imgs):
        corrupted_indices = []

        # Handle batched numpy arrays (N * W * H * 3)
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            n_imgs = len(imgs)
            processed_imgs = []
            for idx, img in enumerate(imgs):
                try:
                    processed_imgs.append(self.preprocess_single_image(img))
                except Exception:
                    corrupted_indices.append(idx)

        # For other types of input (list or single input)
        else:
            if not isinstance(imgs, list):
                imgs = [imgs]
            n_imgs = len(imgs)

            processed_imgs = []
            for idx, img in enumerate(imgs):
                try:
                    processed_imgs.append(self.preprocess_single_image(img))
                except Exception:
                    corrupted_indices.append(idx)

        if corrupted_indices:
            return n_imgs, corrupted_indices

        return n_imgs, processed_imgs

    

    def embed_images(self, imgs, num_workers=1, batch_size=32, normalize=True):
        number_of_images, processed_imgs_or_errors = self.preprocess_images(imgs)
        
        if isinstance(processed_imgs_or_errors[0], int):  # It's an error list
            if number_of_images == 1:
                raise ValueError("The image is inaccessible or corrupted")
            else:
                raise ValueError(f"Some images were inaccessible or corrupted. Indices: {', '.join(map(str, processed_imgs_or_errors))}")
            return
        else:
            list_of_preprocessed_arrays = processed_imgs_or_errors

        dataset = ImageArrayDataset(list_of_preprocessed_arrays)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        image_embeddings = []
        total = len(list_of_preprocessed_arrays) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for images in dataloader:
                if isinstance(images, list):
                    images = torch.cat(images)
                images = images.to(self.device)
                embedding = self.model.get_image_features(images).detach().cpu().numpy()
                image_embeddings.extend(embedding)
                pbar.update(1)
            pbar.close()
        image_embeddings = np.array(image_embeddings)
        # Assuming normalize_embedding is a member variable you want to add, else remove this
        if normalize:
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        return image_embeddings
    

    def embed_texts(self, list_of_labels, num_workers=1, batch_size=32, normalize=True):
        # If the input is a single string, convert it into a list
        if isinstance(list_of_labels, str):
            list_of_labels = [list_of_labels]
            
        dataset = CaptioningDataset(list_of_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        text_embeddings = []
        total = len(list_of_labels) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for captions in dataloader:
                tkn = clip.tokenize(captions, truncate=True).to(self.device)
                embedding = self.model.get_text_features(tkn).detach().cpu().numpy()
                text_embeddings.extend(embedding)
                pbar.update(1)
            pbar.close()
        text_embeddings = np.array(text_embeddings)
        if normalize:
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return text_embeddings
