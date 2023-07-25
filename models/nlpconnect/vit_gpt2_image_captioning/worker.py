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
from share.constant import *
from share.my_image import MyImage
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


# %% ---- 2023-07-06 ------------------------
# Function and class
class CaptionWorker(object):
    worker_name = 'nlpconnect/vit-gpt2-image-captioning'
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning", **cuda_kwargs)
    feature_extractor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning", **cuda_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning", **cuda_kwargs)
    device = cuda_kwargs['device_map']
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def __init__(self):
        print('-' * 40)
        print(self.worker_name)
        print(self.model)
        print(self.feature_extractor)
        print(self.tokenizer)

    def process(self, mi: MyImage):
        tic = time.time()

        pixel_values = self.feature_extractor(
            mi.image['img'], return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        toc = time.time()
        print('Process costs {:.4f} seconds, by {}'.format(
            toc - tic, self.worker_name))

        return preds


# %% ---- 2023-07-06 ------------------------
# Play ground


# %% ---- 2023-07-06 ------------------------
# Pending


# %% ---- 2023-07-06 ------------------------
# Pending
