"""
File: worker.py
Author: Chuncheng Zhang
Date: 2023-07-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-07-06 ------------------------
# Requirements and constants
import io
import numpy as np
from share.constant import *
from share.my_image import MyImage, Image
from transformers import DetrFeatureExtractor, DetrForSegmentation
# from transformers.models.detr.feature_extraction_detr import rgb_to_id
from transformers.image_transforms import rgb_to_id


# %% ---- 2023-07-06 ------------------------
# Function and class
class DETRWorker(object):
    worker_name = 'facebook/detr-resnet-50-panoptic'
    # The device_map like cuda:0 doesn't support the DetrForSegmentation,
    # but I don't know why.
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "facebook/detr-resnet-50-panoptic")  # , **cuda_kwargs)
    model = DetrForSegmentation.from_pretrained(
        "facebook/detr-resnet-50-panoptic")  # , **cuda_kwargs)

    def __init__(self):
        print('-' * 40)
        print(self.worker_name)
        print(self.model)
        print(self.feature_extractor)

    def process(self, mi: MyImage):
        tic = time.time()

        # prepare image for the model
        inputs = self.feature_extractor(
            images=mi.image['img'], return_tensors="pt")

        # forward pass
        outputs = self.model(**inputs)

        import torch
        # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
        processed_sizes = torch.as_tensor(
            inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        # processed_sizes = [1]
        result = self.feature_extractor.post_process_panoptic(
            outputs, processed_sizes)[0]

        # the segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        # retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb_to_id(panoptic_seg)

        toc = time.time()
        print('Process costs {:.4f} seconds, by {}'.format(
            toc - tic, self.worker_name))

        return result, panoptic_seg, panoptic_seg_id


# %% ---- 2023-07-06 ------------------------
# Play ground


# %% ---- 2023-07-06 ------------------------
# Pending


# %% ---- 2023-07-06 ------------------------
# Pending
