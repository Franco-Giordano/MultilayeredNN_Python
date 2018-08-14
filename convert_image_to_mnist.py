from PIL import Image
import numpy as np


pic = Image.open("1.png")
num = np.array(pic)
nums = np.matmul(np.array(pic), np.identity(28))

num = num.resize((784,1))
print(num)
print(nums)


