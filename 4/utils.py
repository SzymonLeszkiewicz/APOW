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
    plt.figure(figsize=(7, 5))
    plt.plot(hist)
    plt.title('Hist gray')
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


def print_info(img):
    print(np.info(img))
    if len(img.shape) == 2:
        print('min/max', img.min(), img.max())
    elif len(img.shape) == 3:
        print('CHANNEL1 min/max', img[:, :, 0].min(), img[:, :, 0].max())
        print('CHANNEL2 min/max', img[:, :, 1].min(), img[:, :, 1].max())
        print('CHANNEL3 min/max', img[:, :, 2].min(), img[:, :, 2].max())


if __name__ == "__main__":
    pass

# %%
