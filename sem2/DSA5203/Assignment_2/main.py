import numpy as np
import cv2


def imrect(im1):
    # Perform Image rectification on an 3D array im.
    # Parameters: im1: numpy.ndarray, an array with H*W*C representing image.
    # (H,W is the image size and C is the channel)
    # Returns: out: numpy.ndarray, rectified image。
    #   out =im1

    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200)
    ret, thresh = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours_thresh, hierarchy_thresh = cv2.findContours(image=thresh,
                                                         mode=cv2.RETR_TREE,
                                                         method=cv2.CHAIN_APPROX_NONE)
    contours_edges, hierarchy_edges = cv2.findContours(image=edges.copy(),
                                                       mode=cv2.RETR_TREE,
                                                       method=cv2.CHAIN_APPROX_NONE)
    contours = contours_edges + contours_thresh

    biggest = biggest_contour(contours)
    rect = order_points(biggest)
    out = warp_image(im1, rect)
    return out


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if 20 < area < 2000000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest


def order_points(pts):
    pts = pts.reshape(4, 2)
    out_pts = np.zeros((4, 2), dtype='float32')

    pts_sum = pts.sum(axis=1)
    out_pts[0] = pts[np.argmin(pts_sum)]
    out_pts[3] = pts[np.argmax(pts_sum)]
    pts = np.delete(pts, np.argmin(pts.sum(axis=1)), 0)
    pts = np.delete(pts, np.argmax(pts.sum(axis=1)), 0)

    pts_diff = np.diff(pts, axis=1)
    out_pts[1] = pts[np.argmin(pts_diff)]
    out_pts[2] = pts[np.argmax(pts_diff)]

    return out_pts


def warp_image(orig, rect):
    (tl, tr, bl, br) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the minimum of the width and height values to reach our final dimensions
    minWidth = min(int(widthA), int(widthB))
    minHeight = min(int(heightA), int(heightB))

    # construct our destination points
    dst = np.array([[100, 100],
                    [minWidth + 99, 100],
                    [100, minHeight + 99],
                    [minWidth + 99, minHeight + 99]],
                   dtype="float32")

    # calculate the perspective transform matrix and warp the perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (orig.shape[1], orig.shape[0]))
    return warp


if __name__ == "__main__":

    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = np.array(cv2.imread(name, -1))
        rectificated = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:], rectificated)
