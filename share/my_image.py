"""
File: my_image.py
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
import time
import hashlib
import requests
import traceback
from PIL import Image
from pathlib import Path


# %% ---- 2023-07-06 ------------------------
# Function and class

class MyImage(object):
    """The base class for the image

    Load the image by the following methods:
    - from_local
    - from_url
    - from_PIL
    - from_bytes

    Use the self.image as the img information.
    """

    def __init__(self):
        self.image = None
        pass

    def compute_img_everything(self, img: Image):
        """Compute all the information from the img object

        Args:
            img (Image): The input img object

        Returns:
            dict: The information of the img object
        """
        # Necessary checks
        if img is None:
            return None

        assert isinstance(
            img, Image.Image), 'The [img] must be an Image instance'

        # Constant
        format = 'jpeg'
        ext = 'jpg'
        mode = 'RGB'
        create_time = time.time()

        # Convert into RGB format
        if img.mode != mode:
            img = img.convert(mode=mode)

        # Write into the BytesIO
        bytes_io = io.BytesIO()
        img.save(bytes_io, format=format)

        # Compute the md5 hash
        md5_hash = hashlib.md5()
        md5_hash.update(bytes_io.getvalue())

        return dict(
            img=img,
            # ---------------------------------
            ext=ext,
            mode=mode,
            format=format,
            create_time=create_time,
            # ---------------------------------
            bytes_io=bytes_io,
            md5_hash=md5_hash,
            get_bytes=bytes_io.getvalue,
            get_hexdigest=md5_hash.hexdigest,
            # ---------------------------------
            unique_id=md5_hash.hexdigest(),
            unique_fname='{}.{}'.format(md5_hash.hexdigest(), ext)
        )

    def from_local(self, path: Path):
        """Init the image from a local image path

        Args:
            path (Path): The local image path

        Returns:
            dict: The img info of the image
        """
        try:
            self.image = None
            img = Image.open(Path(path))
            self.image = self.compute_img_everything(img)
            print('Created image: {}'.format(self.image))
        except:
            traceback.print_exc()
        return self

    def from_url(self, url: str):
        """Init the image from a url

        Args:
            url (str): The remote url of the image

        Returns:
            dict: The img info of the image
        """
        try:
            self.image = None
            img = Image.open(requests.get(url, stream=True).raw)
            self.image = self.compute_img_everything(img)
            print('Created image: {}'.format(self.image))
        except:
            traceback.print_exc()
        return self

    def from_PIL(self, img: Image):
        """Init the image from a Image object

        Args:
            img (Image): The Image object

        Returns:
            dict: The img info of the image
        """
        try:
            self.image = None
            self.image = self.compute_img_everything(img)
            print('Created image: {}'.format(self.image))
            return self.image
        except:
            traceback.print_exc()
        return self

    def from_bytes(self, raw: bytes):
        """Init the image from the raw bytes

        Args:
            raw (bytes): The bytes of the image

        Returns:
            dict: The img info of the image
        """

        try:
            self.image = None
            img = Image.open(raw)
            self.image = self.compute_img_everything(img)
            print('Created image: {}'.format(self.image))
            return self.image
        except:
            traceback.print_exc()
        return self


# %% ---- 2023-07-06 ------------------------
# Play ground


# %% ---- 2023-07-06 ------------------------
# Pending


# %% ---- 2023-07-06 ------------------------
# Pending

# %%
