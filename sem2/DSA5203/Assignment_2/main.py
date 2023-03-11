import numpy as np
import cv2

def imrect(im1):
# Perform Image rectification on an 3D array im.
# Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
# Returns: out: numpy.ndarray, rectified image。
#   out =im1

    return (out)
 

if __name__ == "__main__":

    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = np.array(cv2.imread(name, -1), dtype=np.float32)/255.
        rectificated = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:],np.uint8(np.clip(np.around(rectificated*255,decimals=0),0,255)))
