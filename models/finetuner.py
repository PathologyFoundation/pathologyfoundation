import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import *
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
opj = os.path.join


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ImageDataset_with_label(torch.utils.data.Dataset):
    def __init__(self,
                 image_list,
                 label_list,
                 preprocess=None):
        self.image_list = image_list
        self.label_list = label_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_list[idx]
        label = self.label_list[idx]
        image = Image.open(image_path).convert("RGB")
        if self.preprocess is not None:
            image = self.preprocess(image)
        return image, label
    

class BackboneProber(nn.Module):
    def __init__(self, backbone, classifier):
        super(BackboneProber, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = x.to(self.backbone.vision_model.encoder.layers[0].mlp.fc1.weight.dtype)
        embedding = self.backbone.get_image_features(x) # get embedding via Huggingface approach
        out = self.classifier(embedding)
        out = out.squeeze()
        return out, embedding
    

class FineTuner():
    def __init__(self,
                 backbone,
                 preprocess,
                 num_classes,
                 freeze_vit=False,
                 random_state=42,
                 checkpoint=None,
                 device=None
                 ):
        super(FineTuner, self).__init__()
        backbone = backbone
        self.preprocess = preprocess
        self.num_classes = num_classes
        self.freeze_vit = freeze_vit
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_func = torch.nn.CrossEntropyLoss()  # this is for regression mean squared loss

        self.in_features = backbone.visual_projection.out_features
        classifier = nn.Linear(in_features = self.in_features,
                                    out_features = self.num_classes)
        self.model = BackboneProber(backbone, classifier)
        if checkpoint is not None:
            self.load_model(checkpoint)
        self.train_init()
    
    def predict(self, image):
        # image can be either PIL Image or a list of PIL Image
        x = self.preprocess(image)
        # Convert the list of NumPy arrays to a PyTorch tensor
        tensor_list = [torch.tensor(arr) for arr in x]
        # Convert the list of PyTorch tensors to a single tensor
        x = torch.stack(tensor_list)
        out, _ = self.model(x)
        return out
    
    def extract_embedding(self, image):
        # image can be either PIL Image or a list of PIL Image
        x = self.preprocess(image)
        # Convert the list of NumPy arrays to a PyTorch tensor
        tensor_list = [torch.tensor(arr) for arr in x]
        # Convert the list of PyTorch tensors to a single tensor
        x = torch.stack(tensor_list)
        _, embedding = self.model(x)
        return embedding
        
    def train_init(self):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self,
              df,
              validation_split=0.1,
              batch_size=32,
              num_workers=1,
              lr=1e-5,
              nepochs=100
              ):
        train_loader, test_loader = self.prepare_dataloader(df, validation_split, batch_size, num_workers)
        self.train_nn(train_loader,
                      test_loader,
                      lr=lr,
                      nepochs=nepochs
                      )

    def prepare_dataloader(self,
                           df,
                           validation_split,
                           batch_size,
                           num_workers):

        image_list = np.array(df['image'].to_list())
        label_list  = np.array(df['label'].to_list())

        
        np.random.seed(self.random_state)
        order = np.arange(len(image_list))
        np.random.shuffle(order)
        
        image_list_shuffled = image_list[order]
        label_list_shuffled = label_list[order]
        
        print("Preparing dataloader ...")
        train_idx = order[0:int(len(order)*(1-validation_split))]
        image_dataset = ImageDataset_with_label(image_list_shuffled[train_idx], label_list_shuffled[train_idx], preprocess=self.preprocess)
        train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        if validation_split > 0:
            test_idx = order[int(len(order)*(1-validation_split)):]
            image_dataset = ImageDataset_with_label(image_list_shuffled[test_idx], label_list_shuffled[test_idx], preprocess=self.preprocess)
            test_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            test_loader = None

        return train_loader, test_loader
    

    def train_nn(self,
                 train_loader,
                 test_loader=None,
                 evaluation_step=1,
                 lr=1e-5,
                 nepochs=5,
                 ):
        self.model = self.model.to(self.device)

        if self.freeze_vit:
            optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for epoch in range(nepochs):
            loss_all = 0
            y_true = np.array([])
            y_pred = np.array([])
            loss_log = AverageMeter()
            loss = 0
            tqdm_loader = tqdm(train_loader)
            for x, y in tqdm_loader:
                tqdm_loader.set_description(f"Last Batch -- loss: {loss:.2f}")
                x = torch.cat(x)
                x = x.to(self.device)
                y = y.reshape(-1).to(self.device)
                self.model.train()
                prediction, _ = self.model(x)
                loss = self.loss_func(prediction, y)     # must be (1. nn output, 2. target)
                proba = F.softmax(prediction, dim=1)
                y_hat = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                y_true = np.concatenate((y_true, y.data.cpu().numpy().reshape(-1)))
                y_pred = np.concatenate((y_pred, y_hat.cpu().numpy().reshape(-1)))
                loss_log.update(loss)
                loss_all += loss.data.cpu().numpy()
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            
            acc_train = accuracy_score(y_true, y_pred)
            f1_train = f1_score(y_true, y_pred, average='weighted')
            res_dict_train = {'loss': loss_all, 'accuracy': acc_train, 'f1_weighted': f1_train}
            row_train = pd.DataFrame(res_dict_train, index=[epoch])
            train_df = pd.concat([train_df, row_train], axis=0)
            print(f'epoch: {epoch+1:04d} \t train loss: {loss_all:.4e} \t ACC_train: {acc_train:.2f} \t F1_train: {f1_train:.2f}')

            if (epoch+1) % evaluation_step == 0 and test_loader is not None:
                self.model.eval()
                # plot and show learning process
                y_val_true, y_val_pred, loss_val_all = self.test_nn(test_loader)
                acc_val = accuracy_score(y_val_true, y_val_pred)
                f1_val = f1_score(y_val_true, y_val_pred, average='weighted')
                res_dict_val = {'loss': loss_val_all, 'accuracy': acc_val, 'f1_weighted': f1_val}
                row_test = pd.DataFrame(res_dict_val, index=[epoch])
                test_df = pd.concat([test_df, row_test], axis=0)
                print(f'epoch: {epoch+1:04d} \t validation loss: {loss_val_all:.4e} \t ACC_val: {acc_val:.2f} \t F1_val: {f1_val:.2f}')
        return
        
    def test_nn(self,
                test_loader,
                progress=True,
                calc_loss=True,
                ):
        self.model.eval()
        self.model = self.model.to(self.device)
        
        with torch.no_grad():
            loss_all = 0
            y_true = np.array([])
            y_pred = np.array([])
            if progress:
                pbar = tqdm(total=len(test_loader))
            for x, y in test_loader:
                x = torch.cat(x)
                x = x.to(self.device)
                y = y.reshape(-1).to(self.device)
                if progress:
                    pbar.update()
                
                prediction, _ = self.model(x)
                proba = F.softmax(prediction, dim=1)
                y_hat = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                y_true = np.concatenate((y_true, y.data.cpu().numpy().reshape(-1)))
                y_pred = np.concatenate((y_pred, y_hat.data.cpu().numpy().reshape(-1)))
                if calc_loss:
                    loss = self.loss_func(prediction, y)     # must be (1. nn output, 2. target)
                    loss_all += loss.data.cpu().numpy()
                #print(pd.DataFrame(np.c_[y_true, y_pred], columns=['y_true', 'y_pred']))
        return y_true, y_pred, loss_all
    
    def save_model(self, savedir):
        torch.save(self.model.cpu().state_dict(), savedir)

    def load_model(self, checkpoint):
        ckpt = torch.load(checkpoint)
        ckpt_backbone = OrderedDict((key.replace('backbone.', ''), value) for key, value in ckpt.items() if key.startswith('backbone.'))
        ckpt_classifier = OrderedDict((key.replace('classifier.', ''), value) for key, value in ckpt.items() if key.startswith('classifier.'))
        self.model.backbone.load_state_dict(ckpt_backbone)
        self.model.classifier.load_state_dict(ckpt_classifier)

    