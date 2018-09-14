import numpy as np
import PIL

def import_image(path, dimension, resize = False):
    pic = PIL.Image.open(path).convert('L')
    
    if resize:
        pic = pic.resize(dimension, PIL.Image.ANTIALIAS)
    
    num = np.array(pic)
        
    num.resize((dimension[0] * dimension[1],1))
    
    num = [[u/255 for u in n] for n in num]
    num = np.array(num)
    
    return num
    
