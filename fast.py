import cv2 as cv

"""
fast_detector(image_name, threshold, nonmax_suppression) -> None
.   @brief Generate image with corners defected.
.
.   Function implements FAST algorithm using OpenCV and 'cv2' library (Python wrapper for OpenCV).
.   
.   @param image_name Name of original image
.   @param threshold Threshold on difference between intensity of the central pixel and pixels of circle around this pixel
.   @param nonmax_suppression if 0, non-maximum supression is applied to detected keypoints(corners). Default value is 1
"""


def fast_detector(image_name, threshold, nonmax_suppression=1):
    image_path = 'img/' + image_name + '.jpg'
    img = cv.imread(image_path, 0)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # Configure parameters
    fast.setThreshold(threshold)
    fast.setNonmaxSuppression(nonmax_suppression)

    # Find and draw the keypoints in output file
    kp = fast.detect(img, None)
    img_out = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    output_path = 'img/results/' + image_name + '_%d_%d_.png' % (threshold, nonmax_suppression)
    cv.imwrite(output_path, img_out)

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints: {}".format(len(kp)))


def main():
    image_name = 'SquareBuilding'
    fast_detector(image_name, 10, 1)


if __name__ == '__main__':
    main()
