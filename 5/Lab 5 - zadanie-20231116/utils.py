import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_img_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('RGB')
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_img)
    plt.title('HSV')
    plt.show()


def display_images(images, titles=None, gray=False, figsize=(25, 25)):
    if len(images) == 1:
        plt.figure(figsize=(20, 20))
        if gray:
            plt.imshow(images[0], cmap='gray')
        else:
            plt.imshow(images[0])
        plt.axis('off')
        plt.show()
    else:
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        for i, img in enumerate(images):
            if gray:
                axes[i].imshow(img, cmap='gray')
                try:
                    axes[i].set_title(titles[i])
                except IndexError:
                    print('IndexError')
                except TypeError:
                    print('TypeError')
                except Exception as e:
                    print(e)
            else:
                axes[i].imshow(img)
            axes[i].axis('off')
        plt.show()


def hist_rgb(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].plot(hist)
    axes[0].set_title('Red')
    hist = cv2.calcHist([img], [1], None, [256], [0, 256])
    axes[1].plot(hist)
    axes[1].set_title('Green')
    hist = cv2.calcHist([img], [2], None, [256], [0, 256])
    axes[2].plot(hist)
    axes[2].set_title('Blue')
    plt.show()


def build_bimodal(size, mean1, mean2, sigma1, sigma2, ratio=0.5):
    x = np.random.normal(mean1, sigma1, int(size * ratio))
    y = np.random.normal(mean2, sigma2, int(size * (1 - ratio)))
    return np.concatenate((x, y))


def plot_rgb(img):
    # utwórz obraz skladowych rgb
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    # wysiwetl obraz skladowych rgb w kolorze
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap='Reds')
    plt.title('Red')
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='Greens')
    plt.title('Green')
    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='Blues')
    plt.title('Blue')
    plt.show()


def hist_hsv(hsv_img):
    h = hsv_img[:, :, 0].flatten()
    s = hsv_img[:, :, 1].flatten()
    v = hsv_img[:, :, 2].flatten()
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].hist(h, bins=100)
    axes[0].set_title('Hue')
    axes[1].hist(s, bins=100)
    axes[1].set_title('Saturation')
    axes[2].hist(v, bins=100)
    axes[2].set_title('Value')
    plt.show()


def hist_hsv_normalized(hsv_img):
    hist_h = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
    # normalize
    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].plot(hist_h)
    axes[0].set_title('Hue')
    axes[1].plot(hist_s)
    axes[1].set_title('Saturation')
    axes[2].plot(hist_v)
    axes[2].set_title('Value')
    plt.show()


def normalized_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(figsize=(5, 5))
    plt.plot(hist)
    plt.title('Normalized hist')
    plt.show()


def hist_binary(img):
    binary = img.flatten()
    x = ['0', '255']
    y = [binary.tolist().count(0), binary.tolist().count(255)]
    # normalize y
    y = [i / len(binary) for i in y]
    plt.figure(figsize=(5, 5))
    plt.bar(x, y)
    plt.title('Hist binary')
    # add data label
    for a, b in zip(x, y):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.show()


def hist_gray_plot(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.figure(figsize=(15, 5))
    plt.plot(hist)
    plt.title('Hist gray')
    x_ticks = np.arange(0, 256, 10)
    plt.xticks(x_ticks)
    plt.show()


def show_hsv(hsv_img):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(hsv_img[:, :, 0], cmap='hsv')
    plt.title('Hue')
    plt.subplot(1, 3, 2)
    plt.imshow(hsv_img[:, :, 1], cmap='gray')
    plt.title('Saturation')
    plt.subplot(1, 3, 3)
    plt.imshow(hsv_img[:, :, 2], cmap='gray')
    plt.title('Value')
    plt.show()


def show_lab(lab_img):
    l, a, b = cv2.split(lab_img)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(l, cmap='gray')
    plt.title('L')
    plt.subplot(1, 3, 2)
    plt.imshow(a, cmap='gray')
    plt.title('A')
    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='gray')
    plt.title('B')
    plt.show()


def hist_lab(lab_img):
    l = lab_img[:, :, 0].flatten()
    a = lab_img[:, :, 1].flatten()
    b = lab_img[:, :, 2].flatten()
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].hist(l, bins=100)
    axes[0].set_title('L')
    axes[1].hist(a, bins=100)
    axes[1].set_title('A')
    axes[2].hist(b, bins=100)
    axes[2].set_title('B')
    plt.show()


def hist_lab_normalized(lab_img):
    l, a, b = cv2.split(lab_img)
    hist_l = cv2.calcHist([l], [0], None, [256], [0, 256])
    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    cv2.normalize(hist_l, hist_l, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].plot(hist_l)
    axes[0].set_title('L')
    axes[1].plot(hist_a)
    axes[1].set_title('A')
    axes[2].plot(hist_b)
    axes[2].set_title('B')
    plt.show()


def analyze_channels(img, img_lab, img_hsv):
    plot_rgb(img)
    hist_rgb(img)
    show_hsv(img_hsv)
    hist_hsv(img_hsv)
    show_lab(img_lab)
    hist_lab_normalized(img_lab)


def print_info(img):
    print(np.info(img))
    if len(img.shape) == 2:
        print('min/max', img.min(), img.max())
    elif len(img.shape) == 3:
        print('CHANNEL1 min/max', img[:, :, 0].min(), img[:, :, 0].max())
        print('CHANNEL2 min/max', img[:, :, 1].min(), img[:, :, 1].max())
        print('CHANNEL3 min/max', img[:, :, 2].min(), img[:, :, 2].max())


def detect_biox(img_lab, kernel_size=3, iterations=1, thresh=180):
    l, a, b = cv2.split(img_lab)
    ret, thresh = cv2.threshold(b, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    display_images([thresh, erosion], ['binary', 'erosion'], gray=True)


if __name__ == "__main__":
    pass

#%%
