from typing import Any
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageLoader:
    def __init__(self,
                 image_folder,
                 image_processor,
                 image_aspect_ratio: str = 'square'):
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        assert self.image_aspect_ratio in ['square']

    def __call__(self, image_file) -> Any:
        image_folder = self.image_folder
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        # assume no pad here
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image