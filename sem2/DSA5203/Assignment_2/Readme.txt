Image to be rectified:          test1.jpg,   test2.jpg
Expected rectified image:        result1.jpg result2.jpg


Remark1: The rectified images can be different in terms of heading-direction. As long as the object of interest is showed with its boundaries lies on horizontal and vertical direction, it is considered as a valid result. 

Remark2: the resolution of rectified images can varies between the range of 0.5--1.5

Remark3: the rectified image can either contain the whole content of the original image with black areas, or contain the content of the original image without black areas. However, the rectified image cannot is strongly discouraged to only contain the object of the interest (which is not defined as image rectification)


Get started:

We recommend to use Python with Anaconda3 to manage the Python environment. Here's how to create a clean environment and install library dependencies that may used in this project.

conda create -n hw2 python=3.10
conda activate hw2
python -m pip install --upgrade pip
pip install -r requirements.txt