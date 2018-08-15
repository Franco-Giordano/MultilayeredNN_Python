import numpy as np
from PIL import Image

def import_image(path):
    pic = Image.open(path).convert('L')
    num = np.array(pic)
    
    num.resize((784,1))
    
    num = [[u/255 for u in n] for n in num]
    num = np.array(num)
    
    return num
    
