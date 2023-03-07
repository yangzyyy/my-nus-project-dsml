import numpy as np
import scipy
from PIL import Image
import argparse


def normalize_image_array(arr, size):
    # Normalizing image array to 0-255 range for plotting.
    # Parameters:
    #   arr: ndarray.    An array representing image
    #   size: integer.   An integer representing the half of the array's size
    #  Returns:
    #   out: ndarray.   A normalized array ready for plotting. Same dimension as the input arr.
    arr[:size, :size] = arr[:size, :size] / arr[:size, :size].max() * 255
    arr[:size, size:] = (arr[:size, size:] - arr[:size, size:].min()) / np.ptp(arr[:size, size:]) * 255
    arr[size:, :size] = (arr[size:, :size] - arr[size:, :size].min()) / np.ptp(arr[size:, :size]) * 255
    arr[size:, size:] = (arr[size:, size:] - arr[size:, size:].min()) / np.ptp(arr[size:, size:]) * 255
    return arr


def down_sampling(out, arr, h, g1, g2, g3):
    # Down-sampling input arr based on four Haar wavelet transform kernels h, g1, g2, and g3,
    # and storing the results in one array out.
    # Parameters:
    #   out: ndarray.    An array of zeros that has same size as arr
    #   arr: ndarray.    An image array or coefficients array
    #   h: ndarray.      The low pass filter
    #   g1: ndarray.     The horizontal high pass filter
    #   g2: ndarray.     The vertical high pass filter
    #   g3: ndarray.     The diagonal high pass filter
    #  Returns:
    #   out: ndarray.   An array representing Haar wavelet coefficients. It has the same shape as input arr.

    size = int(arr.shape[0] / 2)
    out[:size, :size] = scipy.signal.convolve2d(arr, np.flip(h))[1::2, 1::2]
    out[:size, size:] = scipy.signal.convolve2d(arr, np.flip(g1))[1::2, 1::2]
    out[size:, :size] = scipy.signal.convolve2d(arr, np.flip(g2))[1::2, 1::2]
    out[size:, size:] = scipy.signal.convolve2d(arr, np.flip(g3))[1::2, 1::2]
    return out


def up_sampling(temp, coef, h, g1, g2, g3):
    # Up-sampling input coef based on four Haar wavelet transform kernels h, g1, g2, and g3.
    # Parameters:
    #   temp: ndarray.    An array of zeros
    #   coef: ndarray.    An array of coefficients
    #   h: ndarray.      The low pass filter
    #   g1: ndarray.     The horizontal high pass filter
    #   g2: ndarray.     The vertical high pass filter
    #   g3: ndarray.     The diagonal high pass filter
    #  Returns:
    #   out: ndarray.   An array representing the image reconstructed from Haar wavelet coefficients.
    #                   It has the same size as temp.

    size = int(temp.shape[0])
    temp[::2, ::2] = coef[:int(size/2), :int(size/2)]
    out = scipy.signal.convolve2d(temp, h)[:-1, :-1]
    temp[::2, ::2] = coef[:int(size/2), int(size/2):size]
    out += scipy.signal.convolve2d(temp, g1)[:-1, :-1]
    temp[::2, ::2] = coef[int(size/2):size, :int(size/2)]
    out += scipy.signal.convolve2d(temp, g2)[:-1, :-1]
    temp[::2, ::2] = coef[int(size/2):size, int(size/2):size]
    out += scipy.signal.convolve2d(temp, g3)[:-1, :-1]
    return out


def haar2d(im, lvl):
    # Computing 2D discrete Haar wavelet transform of a given ndarray im.
    # Parameters:
    #   im: ndarray.    An array representing image
    #   lvl: integer.   An integer representing the level of wavelet decomposition
    #  Returns:
    #   out: ndarray.   An array representing Haar wavelet coefficients with lvl level. It has the same shape as im

    size = im.shape[0]
    out = np.zeros(img.shape)
    global out_norm  # normalized array for plotting decomposition result
    out_norm = np.zeros(img.shape)
    for level in range(lvl):
        temp = np.zeros((size, size))
        size = int(size / 2)
        if level == 0:
            temp = down_sampling(temp, img, h, g1, g2, g3)
            temp_norm = normalize_image_array(np.copy(temp), size)
            out = np.copy(temp)
            out_norm = np.copy(temp_norm)
        else:
            temp = down_sampling(temp, out[:size * 2, :size * 2], h, g1, g2, g3)
            temp_norm = normalize_image_array(np.copy(temp), size)
            out[:size * 2, :size * 2] = np.copy(temp)
            out_norm[:size * 2, :size * 2] = np.copy(temp_norm)
    return out


def ihaar2d(coef,lvl):
    # Computing an image in the form of ndarray from the ndarray coef which represents its DWT coefficients.
    # Parameters:
    #   coef: ndarray   An array representing 2D Haar wavelet coefficients
    #   lvl: integer.   An integer representing the level of wavelet decomposition
    #  Returns:
    #   out: ndarray.   An array representing the image reconstructed from its Haar wavelet coefficients.

    size = coef.shape[0] / (2 ** lvl)
    for level in range(lvl):
        size = int(size * 2)
        temp = np.zeros((size, size))
        out = up_sampling(temp, coef, h, g1, g2, g3)
        coef[:size, :size] = np.copy(out)
    out = out.astype(np.uint8)
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
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    img_path = opt.img_path  # The test image path
    level = opt.level  # The level of wavelet decomposition
    save_pth = opt.save_pth

    # low pass filter
    h = 1/2 * np.array([[1, 1], [1, 1]])

    # high pass filters
    g1 = 1/2 * np.array([[1, 1], [-1, -1]])
    g2 = 1/2 * np.array([[1, -1], [1, -1]])
    g3 = 1/2 * np.array([[1, -1], [-1, 1]])

    img = np.array(Image.open(img_path).convert('L'))
    haar2d_coef = haar2d(img, level)
    recovery = Image.fromarray(ihaar2d(haar2d_coef, level), mode='L')
    recovery.save(save_pth)
    np.save('./haar2_coeff.npy', haar2d_coef)

    # visualization of wavelet decomposition
    # Image.fromarray(out_norm.astype(np.uint8), mode='L').show()
    
    