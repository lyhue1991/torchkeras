from pathlib import Path 
from PIL import Image 
import os

path = Path(__file__)

def get_example_image(img_name='park.jpg'):
    'name can be bus.jpg / park.jpg / zidane.jpg'
    img_path = str(path.parent/f"assets/{img_name}")
    assert os.path.exists(img_path), 'img_name can only be bus.jpg / park.jpg / zidane.jpg'
    return Image.open(img_path)

   
    