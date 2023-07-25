"""
File: models.py
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
import os
from rich import print, inspect
from pathlib import Path
from omegaconf import OmegaConf


# %% ---- 2023-07-06 ------------------------
# Function and class


def readme(model):
    """Create a readme string for the model

    Args:
        model (dict): The description of the model

    Returns:
        str: The readme string
    """
    return f'''
# {model['path']}

```json
{model}
```

---

'''


def prepare_model(model, override=False):
    """Prepare the model

    Args:
        model (dict): The description of the model
        override (bool, optional): Whether override the existing files. Defaults to False.
    """

    share_folder = [
        'share',
    ]

    # Fetch the path
    model['local_path'] = model['path'].replace('-', '_').replace('.', '_')
    p = Path('models', model['local_path'])
    p.mkdir(parents=True, exist_ok=True)

    # Create the readme.md if there is not one
    if not p.joinpath('readme.md').is_file() or override:
        with open(p.joinpath('readme.md'), 'w') as f:
            f.writelines(readme(model))

    # Link folders
    for folder in share_folder:
        os.system(f'ln -frvs {folder} {p.as_posix()}')


# %% ---- 2023-07-06 ------------------------
# Play ground

# Define the models
models = [
    dict(
        method='segment',
        path='facebook/detr-resnet-50-panoptic',
        url='https://huggingface.co/facebook/detr-resnet-50-panoptic',
    ),
    dict(
        method='capture',
        path='nlpconnect/vit-gpt2-image-captioning',
        url='https://huggingface.co/nlpconnect/vit-gpt2-image-captioning',
    ),
    dict(
        method='clip',
        path='openai/clip-vit-large-patch14',
        url='https://huggingface.co/openai/clip-vit-large-patch14'
    )
]

# Prepare the models
for model in models:
    print('-' * 40)
    print(model)
    prepare_model(model)
    print()


# Create and save the Omega configuration
oc = OmegaConf.create(models)
OmegaConf.save(oc, 'models.yaml')
print('The folder of the models have been created, see models.yaml for detail.')

# %% ---- 2023-07-06 ------------------------
# Pending

# %% ---- 2023-07-06 ------------------------
# Pending

# %%
