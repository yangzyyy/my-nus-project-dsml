import numpy as np
import scipy
from PIL import Image
import argparse
def haar2d(im,lvl):
# Computing 2D discrete Haar wavelet transform of a given ndarray im.
# Parameters: 
#   im: ndarray.    An array representing image
#   lvl: integer.   An integer representing the level of wavelet decomposition
#  Returns:
#   out: ndarray.   An array representing Haar wavelet coefficients with lvl level. It has the same shape as im

# ----
# Insert your code here
# ----
    return out
 
def ihaar2d(im,lvl):
# Computing an image in the form of ndarray from the ndarray coef which represents its DWT coefficients.
# Parameters: 
#   coef: ndarray   An array representing 2D Haar wavelet coefficients
#   lvl: integer.   An integer representing the level of wavelet decomposition
#  Returns:
#   out: ndarray.   An array representing the image reconstructed from its Haar wavelet coefficients.

# ----
# Insert your code here
# ----
    return out


if __name__ == "__main__":
# Code for testing.
# Please modify the img_path to the path stored image and the level of wavelet decomposition.
# Feel free to revise the codes for your convenience
# If you have any question, please send email to e0444157@u.nus.edu for help
# As the hw_1.pdf mentioned, you can also write the test codes on other .py file.

    parser = argparse.ArgumentParser(description="wavelet")
    parser.add_argument("--img_path",  type=str, default='./test.png',  help='The test image path')
    parser.add_argument("--level", type=int, default=4, help="The level of wavelet decomposition")
    parser.add_argument("--save_pth", type=str, default='./recovery.png', help="The save path of reconstructed image ")
    opt = parser.parse_args()

    img_path = opt.img_path # The test image path
    level = opt.level # The level of wavelet decomposition
    save_pth = opt.save_pth

    img = np.array(Image.open(img_path).convert('L'))
    haar2d_coef = haar2d(img,level)
    recovery =  Image.fromarray(ihaar2d(haar2d_coef,level),mode='L')
    recovery.save(save_pth)
    np.save('./haar2_coeff.npy',haar2d_coef)
    
    