"""
File: example.py
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
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from share.my_image import MyImage
from models.openai.clip_vit_large_patch14.worker import ClipWorker
from models.nlpconnect.vit_gpt2_image_captioning.worker import CaptionWorker
from models.facebook.detr_resnet_50_panoptic.worker import DETRWorker


# %% ---- 2023-07-06 ------------------------
# Function and class
clip_worker = ClipWorker()
caption_worker = CaptionWorker()
detr_worker = DETRWorker()

# %%
mi = MyImage()
mi.from_local('share/image/000000039769.jpg')

mia = MyImage()
mia.from_local('share/image/c.jpg')

mib = MyImage()
mib.from_local('share/image/b.jpg')

# %%
print(clip_worker.process(['Cat', 'Dog', 'Sky'], mi))
print(caption_worker.process(mi))

res, mark3, mark1 = detr_worker.process(mi)
print(res['segments_info'])
print(mark3.shape, mark3.dtype)
print(mark1.shape, mark1.dtype)

# %%


def detr_worker_process(mi):
    res, mark3, mark1 = detr_worker.process(mi)
    print(res['segments_info'])
    print(mark3.shape, mark3.dtype)
    print(mark1.shape, mark1.dtype)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    ax = axs[0]
    im = ax.imshow(mi.image['img'].resize(mark1.shape[::-1]))
    ax.axis('off')
    ax.set_title('Raw')

    ax = axs[1]
    im = ax.imshow(mark1)
    ax.axis('off')
    ax.set_title('Segment')
    # plt.colorbar(im)

    fig.tight_layout()

    return res, fig


# %%
res, fig = detr_worker_process(mi)
plt.show()

res, fig = detr_worker_process(mib)
plt.show()

res, fig = detr_worker_process(mia)
plt.show()
# %% ---- 2023-07-06 ------------------------
# Play ground
np.array(mib.image['img']).shape

# %% ---- 2023-07-06 ------------------------
# Pending

# %% ---- 2023-07-06 ------------------------
# Pending
for r in np.linspace(0, 1.0, 11):
    im = Image.fromarray(
        (np.array(mia.image['img']) * r + np.array(mib.image['img']) * (1-r)).astype(np.uint8))
    mi_interpolate = MyImage()
    mi_interpolate.from_PIL(im)
    res, fig = detr_worker_process(mi_interpolate)

    title = '- {:0.2f} - {}'.format(r, caption_worker.process(mi_interpolate))
    fig.suptitle(title)

    fig.tight_layout()

    fig.savefig('interpolate/{}.jpg'.format(title))

    plt.show()

# %%

# %%
