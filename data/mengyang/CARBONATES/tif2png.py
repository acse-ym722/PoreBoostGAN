from tqdm import tqdm
import os  
import numpy as np  
from skimage import io

def main():
    LR = io.imread('c2_lo_resized_256_768_8bits.tif')
    if not os.path.exists("Low_resolution"):
        os.mkdir("Low_resolution")
    for i in tqdm(range(0, LR.shape[0])):
        io.imsave(f"Low_resolution/{i}.png", LR[i,:,:])
    print('Low resolution images generated')

    HR = io.imread('c2_hi_resized_1024_3072_8bits.tif')
    if not os.path.exists("High_resolution"):
        os.mkdir("High_resolution")
    for i in tqdm(range(0, LR.shape[0])):
        io.imsave(f"High_resolution/{i}.png", HR[i*4+1,:,:])
    print('High resolution images generated')
    

if __name__ == '__main__':
    main()
