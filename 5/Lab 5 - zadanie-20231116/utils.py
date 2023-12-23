#%%
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
        plt.figure(figsize=figsize)
        if gray:
            plt.imshow(images[0], cmap='gray')
        else:
            img = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        plt.axis('off')
        plt.title(titles[0] if titles else '', fontsize=20)
        plt.show()
    else:
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        for i, img in enumerate(images):
            if gray:
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(titles[i] if titles else '')
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].set_title(titles[i] if titles else '', fontsize=20)
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


def analyze_channels(img, img_lab, img_hsv):
    plot_rgb(img)
    hist_rgb(img)
    show_hsv(img_hsv)
    hist_hsv(img_hsv)
    show_lab(img_lab)
    hist_lab(img_lab)

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


def calculate_aspect_ratio(ellipse):
    len1, len2 = ellipse[1]
    return len1 / len2


def get_mask_pills(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    ret1, th1 = cv2.threshold(H, 80, 255, cv2.THRESH_BINARY_INV)
    return th1


def get_mask_espu(img, morph=True, kernel_size=3, iterations=1, thresh_val=180):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    ret, thresh = cv2.threshold(b, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if morph:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return thresh


def blue_xy(img):
    L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    ret1, th1 = cv2.threshold(A, 165, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(th1, kernel, iterations=3)
    close = erode
    # close = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if (len(c) > 5 and cv2.contourArea(c) > 300)]
    blue_xy = []
    for c in contours:
        # feat elipses
        ellipse = cv2.fitEllipse(c)
        x, y = ellipse[0]
        blue_xy.append((x, y))
    return blue_xy


def detect_whites(img):
    img = cv2.GaussianBlur(img, (1, 1), 0)
    mask = get_mask_pills(img)
    result = apply_mask(img, mask)

    L, A, B = cv2.split(cv2.cvtColor(result, cv2.COLOR_BGR2LAB))
    ret1, th1 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY)
    espu = get_mask_espu(img)
    result2 = apply_mask(th1, espu)
    kernel = np.ones((3, 3), np.uint8)
    result2 = cv2.morphologyEx(result2, cv2.MORPH_OPEN, kernel, iterations=11)
    # erode
    kernel = np.ones((3, 3), np.uint8)
    result2 = cv2.erode(result2, kernel, iterations=13)
    # morpho close
    kernel = np.ones((1, 1), np.uint8)
    result2 = cv2.morphologyEx(result2, cv2.MORPH_CLOSE, kernel, iterations=1)
    # result2 = cv2.distanceTransform(result2, cv2.DIST_L2, 5)
    # result2 = cv2.normalize(result2, result2, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return result2


def detect_espu(img_lab, morph=True, kernel_size=3, iterations=1, thresh_val=180):
    l, a, b = cv2.split(img_lab)
    ret, thresh = cv2.threshold(b, thresh_val, 255, cv2.THRESH_BINARY)

    if morph:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
    post = cv2.erode(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(post, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pills = [c for c in contours if cv2.contourArea(c) > 90]

    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    img = cv2.drawContours(img_rgb.copy(), pills, -1, (0, 255, 0), 3)

    for c in pills:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.subplots(1, 1, figsize=(12, 8))[1].imshow(img_rgb)
    plt.show()


def try_morpho(img, kernel_size=3, iterations=1):
    mrpho_op = [cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]
    mrpho_name = ['Open', 'Close', 'Gradient', 'TopHat', 'BlackHat']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=iterations)
    display_images([img, dilate], [f'Original', f'Dilate'], gray=True)
    erode = cv2.erode(img, kernel, iterations=iterations)
    display_images([img, erode], [f'Original', f'Erode'], gray=True)
    for op in enumerate(mrpho_op):
        img_morpho = cv2.morphologyEx(img, op[1], kernel, iterations=iterations)
        display_images([img, img_morpho], [f'Original', f'{mrpho_name[op[0]]}'], gray=True)


def detect_carbon(
        img: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 3,
        thresh_val: int = 60):
    """
    Funkcja wykrywająca tabletki w obrazie
    :param img: obraz wejściowy
    :param kernel_size: rozmiar jądra
    :param iterations: liczba iteracji
    :param thresh_val: wartość progu
    :return: lista obrazów z wykrytymi tabletkami
    """
    L, A, B = cv2.split(img)
    ret, thresh = cv2.threshold(L, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
    plt.subplots(1, 1, figsize=(12, 8))[1].imshow(thresh, cmap='gray')
    plt.show()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pills = [c for c in contours if cv2.contourArea(c) > 4000]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    for c in pills:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.subplots(1, 1, figsize=(12, 8))[1].imshow(img_rgb)
    plt.show()
    return [img_rgb]


def detect_carbon(
        img: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 3,
        thresh_val: int = 60):
    """
    Funkcja wykrywająca tabletki w obrazie
    :param img: obraz wejściowy
    :param kernel_size: rozmiar jądra
    :param iterations: liczba iteracji
    :param thresh_val: wartość progu
    :return: lista obrazów z wykrytymi tabletkami
    """
    L, A, B = cv2.split(img)
    ret, thresh = cv2.threshold(L, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
    # distance transform
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L1, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    display_images([thresh, dist_transform], ['thresh', 'dist_transform'], 1)
    thresh = dist_transform

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pills = [c for c in contours if cv2.contourArea(c) > 4000]
    # draw contours
    img2 = cv2.drawContours(img.copy(), pills, -1, (0, 255, 0), 3)
    display_images([img, img2], ['original', 'contours'], 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    for c in pills:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.subplots(1, 1, figsize=(12, 8))[1].imshow(img_rgb)
    plt.show()
    return [img_rgb]


def get_mask(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(img_lab)
    ret, thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def get_box_contours(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(img_lab)
    mask = get_mask(img)
    result = apply_mask(img, mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box = [c for c in contours if cv2.contourArea(c) > 100000]
    temp = cv2.drawContours(img.copy(), box, -1, (0, 255, 0), 3)
    return box


def crop_box(img, angel=21, w=-50, h=-50):
    box_con = get_box_contours(img)
    rect = cv2.boundingRect(box_con[0])
    image = img
    # Pobierz punkty narożników prostokąta
    x, y, width, height = rect
    temp = cv2.drawContours(image.copy(), box_con, -1, (0, 255, 0), 3)
    rect_pts = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)

    center = np.array([x + width / 2, y + height / 2])

    # Oblicz macierz rotacji
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angel, 1.0)  # 1.0 oznacza brak skalowania
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))  # rotated to obraz po rotacji
    rotated_crop = cv2.getRectSubPix(rotated, (width, height), tuple(center))

    box_con = get_box_contours(rotated_crop)
    rect = cv2.boundingRect(box_con[0])
    image = rotated_crop
    # Pobierz punkty narożników prostokąta
    x, y, width, height = rect
    x -= w
    y -= h
    width += w * 2
    height += h * 2
    rect_pts = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)

    center = np.array([x + width / 2, y + height / 2])
    angel = 0

    # Oblicz macierz rotacji
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angel, 1.0)  # 1.0 oznacza brak skalowania
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))  # rotated to obraz po rotacji

    rotated_crop = cv2.getRectSubPix(rotated, (width, height), tuple(center))
    return rotated_crop


# %%
def detect_keto(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    ret, thresh = cv2.threshold(S, 28, 255, cv2.THRESH_BINARY_INV)
    # morpho open
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8), iterations=2)
    # distans transform
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L1, 3)
    thresh2 = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # erode
    thresh2 = cv2.erode(thresh2, np.ones((3, 3), np.uint8), iterations=7)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    keto = [c for c in contours if cv2.contourArea(c) > 1000]
    return keto
    # # draw contours
    # img = img_bgr.copy()
    # for c in biox:
    #     text = 'KETO'
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1
    #     color = (0, 255, 0)
    #     thickness = 2
    #     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    #     x, y, w, h = cv2.boundingRect(c)
    #     cv2.putText(img, text, (x + w // 2 - text_size[0] // 2, y + h // 2 + text_size[1] // 2), font, font_scale, color, thickness)
    # return img


if __name__ == "__main__":
    pass

# %%
