from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility
import skimage
from skimage import io
from skimage.color import rgb2gray
import os
import numpy as np
import dask.array.image as dimg
import matplotlib.pyplot as plt
import pandas as pd



data = io.imread_collection("*.png")

# for colors in original image
#img_gray = rgb2gray(img)
# for im in data:
#     res = list(set(i for j in im for i in j)) 
#     print ("The original matrix is : " + str(res)) 
grayscaleData = [rgb2gray(img) for img in data]

# for img in grayscaleData:
#     plt.imshow(img)
#     plt.show()

# phaseI = 209
# phaseII = 209
# phaseI = 188
# phaseII = 188
phaseI = 76
phaseII = 76
rMax = 200

def individualRow2Point(img_gray, rMax, row, phaseI, phaseII):
    #rMax usually img_gray.shape[1]-1
    Width = img_gray.shape[1]
    Encounters = [0]*(rMax+1)
    probabilityS = [None]*(rMax+1)
    for r in range (0,rMax+1):
        lMax = Width - r
        for a in range(0,lMax):
            pixelOne = img_gray[row, a]
            pixelTwo = img_gray[row, a+r]
            if (pixelOne==phaseI and pixelTwo==phaseII) or (pixelOne==phaseII and pixelTwo==phaseI):
                Encounters[r] = Encounters[r]+1
        probabilityS[r] = Encounters[r]/lMax
    return probabilityS
#print(probabilityS == individualRow2Point(img_gray, rMax, row, phaseI))
def entireImage2Point(img_gray, rMax, phaseI, phaseII):
    final = np.array([0]*(rMax+1))
    for row in range(0,img_gray.shape[0]):
        #print(row)
        curRow = individualRow2Point(img_gray, rMax, row, phaseI, phaseII)
        final = np.add(final, np.array(curRow))
    
    #return final.tolist()
    entireImageProbabilities = [item / (img_gray.shape[0]*1.0) for item in final.tolist()]
    return entireImageProbabilities


p = []
index = 1
for img in grayscaleData:
    x = entireImage2Point(img, rMax, phaseI, phaseII)
    p.append(x)
    plt.plot(x)
    print(str(index)+" done")
    index = index+1


df = pd.DataFrame({"1" : p[0], "2" : p[1], "3" : p[2], "4" : p[3], "5" : p[4], "6" : p[5]})
df.to_csv("submission76.csv", index=False)
plt.show()


