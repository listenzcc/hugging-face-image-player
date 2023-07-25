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
from transformers import CLIPProcessor, CLIPModel


# %% ---- 2023-07-06 ------------------------
# Function and class


class ClipWorker(object):
    worker_name = 'openai/clip-vit-large-patch14'
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", **cuda_kwargs)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", **cuda_kwargs)

    def __init__(self):
        print('-' * 40)
        print(self.worker_name)
        print('model', self.model)
        print('processor', self.processor)

    def process(self, text: list, mi: MyImage):
        """Process the image with the clipped text list

        Args:
            text (list): The text list to be clipped.
            mi (MyImage): The input image to be processed.

        Returns:
            list: The list of tuple, (probs value, logits value, text element)
        """
        tic = time.time()

        inputs = self.processor(text=text,
                                images=mi.image['img'],
                                return_tensors="pt",
                                padding=True)

        outputs = self.model(**inputs)

        self.outputs = outputs

        logits_per_image = outputs.logits_per_image

        probs = logits_per_image.softmax(
            dim=1).detach().cpu().numpy().squeeze()

        logits = logits_per_image.detach().cpu().numpy().squeeze()

        toc = time.time()
        print('Process costs {:.4f} seconds, by {}'.format(
            toc - tic, self.worker_name))

        return [(a, b, c) for a, b, c in zip(probs, logits, text)]


# %% ---- 2023-07-06 ------------------------
# Play ground


# %% ---- 2023-07-06 ------------------------
# Pending


# %% ---- 2023-07-06 ------------------------
# Pending
