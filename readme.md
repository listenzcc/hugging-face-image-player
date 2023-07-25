# Image Player

---

The Large models workers for images.
See [./models.yaml](./models.yaml) for detail of the models.

```yaml
- method: segment
  path: facebook/detr-resnet-50-panoptic
  url: https://huggingface.co/facebook/detr-resnet-50-panoptic
  local_path: facebook/detr_resnet_50_panoptic
- method: capture
  path: nlpconnect/vit-gpt2-image-captioning
  url: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
  local_path: nlpconnect/vit_gpt2_image_captioning
- method: clip
  path: openai/clip-vit-large-patch14
  url: https://huggingface.co/openai/clip-vit-large-patch14
  local_path: openai/clip_vit_large_patch14
```

## Installation

The [./environment.yml](./environment.yml) contains the required python modules.
It only provides the basic environment,
the user will need to install the modules and adjust their versions to fit the models.

## Initialization folder for the models

The [./models.py](./models.py) is used to initialize the folder for the models.
The folders are created automatically into the folder of [./models/](./models/)

## Public methods and data folders

The [./share](./share) folder is used as the container of the public methods and data folders.
And it is automatically linked to the models' folders using soft-link.