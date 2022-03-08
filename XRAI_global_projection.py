#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : XRAI_global_projection.py
# Modified   : 08.03.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import os 
from tqdm import tqdm  
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
import numpy as np  
import utils_xai as utils
from utils_xai import Net 
import json
import random

import saliency.core as saliency
# %matplotlib inline

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

arch_r = resnet50(pretrained=True) 
arch_ef = EfficientNet.from_pretrained('efficientnet-b2')

model = Net(arch=arch_ef).eval()
model.to(device)

# Register hooks for Grad-CAM, which uses the last convolution layer
conv_layer = model.arch._conv_head
conv_layer_outputs = {}


def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()

#conv_layer.register_forward_hook(conv_layer_forward)
#conv_layer.register_full_backward_hook(conv_layer_backward)

# call_model_function is how we pass inputs to our model and receive outputs necessary to computer saliency masks.

class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = utils.PreprocessImages(images)
    target_class_idx =  call_model_args[class_idx_str]
    output = model(images)
    output = torch.sigmoid(output)
    if target_class_idx == 0:
        output = 1-output
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        # outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(output, images, grad_outputs=torch.ones_like(output))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().cpu().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs


directory = "/workspace/stylegan2-ada-pytorch/processed_dataset_256"
filename = "dataset.json"

# Construct the saliency object. This alone doesn't do anthing.
xrai_object = saliency.XRAI()

# Create XRAIParameters and set the algorithm to fast mode which will produce an approximate result.
xrai_params = saliency.XRAIParameters()
xrai_params.algorithm = 'fast'

images_pil = []
if not os.path.exists('/workspace/Explainability_Dermatology/embeddings.npz'):
    images_masked = []
    metadata = [] 
    embeddings = []
    with open(os.path.join(directory, filename)) as file:
        data = json.load(file)['labels']
        random.shuffle(data)
        data = data[:3000]
        for n, (img, label) in tqdm(enumerate(data)):
            # Load an image and infer 

            # Load the image
            img_dir = os.path.join(directory,img) 
            im_orig = utils.LoadImage(img_dir)
            im_tensor = utils.PreprocessImages([im_orig]).to(device)
            # Infer
            features = model.arch(im_tensor)  # 500D features
            prediction = model.output(features)
            prediction = torch.sigmoid(prediction)
            prediction = torch.tensor([[1-prediction, prediction]], device='cuda:0') 
            prediction = prediction.detach().cpu().numpy()
            prediction_class = np.argmax(prediction[0])
            call_model_args = {class_idx_str: prediction_class}
            im = im_orig.astype(np.float32)

            # Compute XRAI attributions with fast algorithm
            xrai_attributions = xrai_object.GetMask(im, call_model_function, call_model_args, extra_parameters=xrai_params, batch_size=20)

            # Mask the image with the most salient 15% of the image
            mask = xrai_attributions > np.percentile(xrai_attributions, 85)
            im_mask = np.array(im_orig)
            im_mask[~mask] = 0 

            # Save data for projection
            embeddings.append(features.cpu().detach().numpy())
            images_pil.append(transform(Image.open(img_dir).resize((100, 100))))
            images_masked.append(im_mask.flatten())
            metadata.append([label, img])

    np.savez('/workspace/Explainability_Dermatology/embeddings.npz', np.array(embeddings))  
    np.savez('/workspace/Explainability_Dermatology/metadata.npz', np.array(metadata))  
    np.savez('/workspace/Explainability_Dermatology/masked_img.npz', np.array(images_masked))  
else:
    embeddings = np.load('/workspace/Explainability_Dermatology/embeddings.npz')["arr_0"]  # (N,1,500)
    masked_img = np.load('/workspace/Explainability_Dermatology/masked_img.npz')["arr_0"] # (N, 256x256x3)
    metadata = np.load('/workspace/Explainability_Dermatology/metadata.npz')["arr_0"] # (N, 2)
    metadata = [l.tolist() for l in metadata]
    for label, dir in metadata:
        img_dir = os.path.join(directory, dir)
        images_pil.append(transform(Image.open(img_dir).resize((100, 100))))

writer = SummaryWriter('/workspace/Explainability_Dermatology')

writer.add_embedding(
                    np.array(embeddings).squeeze(), 
                    metadata=metadata,
                    metadata_header=["label","image_name"],
                    label_img=torch.stack(images_pil),
                    global_step=1, 
                    ) 

writer.add_embedding(
                    np.array(masked_img).squeeze(), 
                    metadata=metadata,
                    metadata_header=["label","image_name"],
                    label_img=torch.stack(images_pil),
                    global_step=2, 
                    ) 

writer.close() 