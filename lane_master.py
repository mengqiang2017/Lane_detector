import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
print(image.shape)


def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)  # 灰度图像高斯模糊处理


def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)     # 填充多边形内的颜色

    masked_image = cv2.bitwise_and(img, mask)           # 合并图层
    return masked_image





def select_point(k, points, threshold):
    while True:
        k_mean = np.mean(k)
        diff = np.abs(k - k_mean)
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            k.pop(idx)
            points.pop(2*idx)
            points.pop(2*idx + 1)
        else:
            break


def line_fit(points, y_min, y_max):
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    fit_line = np.poly1d(np.polyfit(y, x, 1))
    x_min = int(fit_line(y_min))
    x_max = int(fit_line(y_max))
    return (x_min, y_min), (x_max, y_max)

def drawlines(img, lines):
    k_left = []
    k_right = []
    points_left = []
    points_right = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1)/(x2 - x1)
            if k >= 0:
                k_left.append(k)
                points_left.append([x1, y1])
                points_left.append([x2, y2])
            else:
                k_right.append(k)
                points_right.append([x1, y1])
                points_right.append([x2, y2])

    select_point(k_right, points_right, 0.1)
    select_point(k_left, points_left, 0.1)      # 得到所以满足条件的左右两侧的点

    left_pt_1, left_pt_2 = line_fit(points_left, 320, imshape[0])
    right_pt_1, right_pt_2 = line_fit(points_right, 320, imshape[0])
    cv2.line(img, left_pt_1, left_pt_2, [218, 112, 214], 8)
    cv2.line(img, right_pt_1, right_pt_2, [218, 112, 214], 8)


def hough_line(img, rho, theta, threshold, minLineLength, maxLineGap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
    line_img = np.zeros_like(image)
    drawlines(line_img, lines)
    return line_img




if __name__=="__main__":
    # hough_line parameters
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20

    imshape = image.shape
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)         # 图像灰度处理
    blur_img = gaussian_blur(grayscale, 5)                      # 高斯模糊,kernel_size = (5,5)
    edges = canny(blur_img, 10, 150)                            # Canny edge 检测
    vertices = np.array([[(0, imshape[0]), (400, 320), (800, 400), (imshape[1], imshape[0])]])  # 车道线区域
    masked_img = region_of_interest(edges, vertices)            # 获取仅包含车道线区域的供Hough直线检测的图片
    line_img = hough_line(masked_img, rho, theta, threshold, min_line_length, max_line_gap)
    out_img = cv2.addWeighted(image, 0.8, line_img, 1., 0.)     # 输出检测图片


    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.subplot(2, 3, 2)
    plt.imshow(grayscale, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.imshow(edges)
    plt.subplot(2, 3, 4)
    plt.imshow(masked_img)
    plt.subplot(2, 3, 5)
    plt.imshow(line_img)
    plt.subplot(2, 3, 6)
    plt.imshow(out_img)
    plt.show()
