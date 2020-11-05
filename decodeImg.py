from PIL import Image
import numpy as np 
import os
import time
from threading import Thread

device = 'fd6c34c6-92d3-45b6-ba63-731a691bdbc8'
# directory =  r'/home/james/code/ml-doop/imgs/fd6c34c6-92d3-45b6-ba63-731a691bdbc8/982 123732304708'
topDir =  r'/home/james/code/ml-doop/imgs/'+device

# print(os.listdir(directory))

i = 0
# for filename in os.listdir(directory):
#     # print(filename)
#     print(os.path.join(directory, filename))
#     rgb = np.load(os.path.join(directory, filename))
#     img = Image.fromarray(rgb,'RGB')
#     name = filename+'.png'

#     img.save('/home/james/code/ml-doop/decoded/'+filename+'/'+name)
#     i = i+1

for directory in os.listdir(topDir):
    encodedAnimalPics = topDir+'/'+directory
    saveDir = '/home/james/code/ml-doop/decoded/'+device+'/'+directory
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for animal in os.listdir(encodedAnimalPics):
        # print(animal)  
        rgb = np.load(os.path.join(encodedAnimalPics, animal))
        img = Image.fromarray(rgb,'RGB')
        name = animal+'.png' 
        img.save(saveDir+'/'+name)
    # os.mkdir('/home/james/code/ml-doop/decoded/'+device+'/'+directory)
    # print(directory)