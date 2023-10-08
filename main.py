import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import configuration as config

sys.setrecursionlimit(10**5)

func = lambda x, y, center, sigma: np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)) / (
        2 * np.pi * sigma ** 2)


# ЧБ
def black_and_white(arr):
    X, Y, U = arr.shape
    result = np.zeros((X, Y), dtype=int)
    for i in range(X):
        for j in range(Y):
            result[i, j] = np.mean(arr[i, j])
    return np.array(result, dtype=np.uint8)


# Формирование фильтра Гаусса определённой размерности
def gauss(shape):
    matrix = np.zeros((shape, shape))
    center = shape // 2
    for i in range(shape):
        for j in range(shape):
            matrix[i, j] = func(i, j, center, config.sigma)
    matrix /= np.sum(matrix)
    return matrix


# Наложение фильтра Гаусса
def filter(arr, shape):
    X, Y = arr.shape
    center = shape // 2
    borderX, borderY = arr.shape
    # Добавляем рамку
    borderX += center * 2
    borderY += center * 2
    result = np.zeros((borderX, borderY), dtype=np.uint8)
    # Внутрь помещаем изображение
    result[center:-center, center:-center] = arr
    # Генерим фильтр
    matrix = gauss(shape)
    # Накладываем фильтр
    for i in range(X):
        for j in range(Y):
            result[i + center, j + center] = int(np.sum(result[i:i + shape, j:j + shape] * matrix))
    return result[center:-center, center:-center]


# Магнитуда и направление из оценок частных производных для всей картинки
def gradient(arr):
    X, Y = arr.shape
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Magn, Dir = np.zeros((X, Y)), np.zeros((X, Y))
    for i in range(1, X - 1):
        for j in range(1, Y - 1):
            neighbours = arr[i - 1: i + 2, j - 1: j + 2]
            # Получаем оценки частных производных
            Ix = np.sum(Gx * neighbours)
            Iy = np.sum(Gy * neighbours)
            # По ним оцениваем магнитуду и направление
            magn, dir = magnitude_and_direction(Ix, Iy)
            # Направление округляем до кратности 45 градусов
            dir = round(dir, angles)
            Magn[i][j], Dir[i][j] = magn, dir
    return Magn.astype(int), Dir


# Магнитуда и направление градиента
def magnitude_and_direction(Ix, Iy):
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    direction = np.arctan2(Iy, Ix) * 180 / np.pi
    return magnitude, direction


# Округление направлений
def round(dir, angles):
    if dir < 0:
        dir += 360
    elif dir > 337.5:
        return 0
    return min(angles, key=lambda x: abs(x - dir))


# Подавление немаксимумов
def suppression(magn, dir):
    for i in range(1, magn.shape[0] - 1):
        for j in range(1, magn.shape[1] - 1):
            if dir[i][j] == 0 or 180:
                if magn[i][j] <= magn[i][j - 1] or magn[i][j] <= magn[i][j + 1]:
                    magn[i][j] = 0
            elif dir[i][j] == 90 or 270:
                if magn[i][j] <= magn[i - 1][j] or magn[i][j] <= magn[i + 1][j]:
                    magn[i][j] = 0
            elif dir[i][j] == 45 or 225:
                if magn[i][j] <= magn[i - 1][j + 1] or magn[i][j] <= magn[i + 1][j - 1]:
                    magn[i][j] = 0
            else:
                if magn[i][j] <= magn[i - 1][j - 1] or magn[i][j] <= magn[i + 1][j + 1]:
                    magn[i][j] = 0


# Рекурсивно двигаемся от пикселя для которого 𝑔(𝑝) > 𝑇ℎ𝑖𝑔ℎ
def search(magn, borders, t_low, x, y):
    square = magn[x - 1: x + 2, y - 1: y + 2]
    for m in range(-1, 2):
        for n in range(-1, 2):
            if (m == 0 and n == 0) or (x + m <= 0 or x + m >= magn.shape[0] - 1) or (y + n <= 0 or y + n >= magn.shape[1] - 1):
                continue
            if square[m][n] > t_low and borders[x + m][y + n] == 0:
                borders[x + m][y + n] = 255
                search(magn, borders, t_low, x + m, y + n)
    return


def clarification(magn):
    borders = np.zeros((magn.shape[0], magn.shape[1]), dtype=int)
    for i in range(magn.shape[0] - 1):
        for j in range(1, magn.shape[1] - 1):
            if magn[i][j] > config.t_high and borders[i][j] == 0:
                borders[i][j] = 255
                search(magn, borders, config.t_low, i, j)
    return borders


# Считываем изображения
image = np.array(Image.open("brain21.jpg"))
plt.subplot(1, 3, 1)
plt.imshow(image)
"""
# 1
imageAngel = np.array(Image.open("one.jpg"))
plt.subplot(3, 3, 1)
plt.imshow(imageAngel)
# 2
imageCat = np.array(Image.open("two.jpg"))
plt.subplot(3, 3, 2)
plt.imshow(imageCat)
# 3
imageDog = np.array(Image.open("three.jpg"))
plt.subplot(3, 3, 3)
plt.imshow(imageDog)
"""

# Полутон
image = black_and_white(image)
"""
# 1
imageAngel = black_and_white(imageAngel)
# 2
imageCat = black_and_white(imageCat)
# 3
imageDog = black_and_white(imageDog)
"""

# Гаусс
image = filter(image, 7)
plt.subplot(1, 3, 2)
plt.imshow(image, cmap='gray')
"""
# 1
imageAngel = filter(imageAngel, 5)
plt.subplot(3, 3, 4)
plt.imshow(imageAngel, cmap='gray')
# 2
imageCat = filter(imageCat, 5)
plt.subplot(3, 3, 5)
plt.imshow(imageCat, cmap='gray')
# 3
imageDog = filter(imageDog, 5)
plt.subplot(3, 3, 6)
plt.imshow(imageDog, cmap='gray')
"""

# Градиент с помощью фильтра Собеля
image, dir1 = gradient(image)
plt.subplot(1, 3, 3)
plt.imshow(image, cmap='gray')
plt.show()
"""
# 1
imageAngel, dir1 = gradient(imageAngel)
plt.subplot(3, 3, 7)
plt.imshow(imageAngel, cmap='gray')
# 2
imageCat, dir2 = gradient(imageCat)
plt.subplot(3, 3, 8)
plt.imshow(imageCat, cmap='gray')
# 3
imageDog, dir3 = gradient(imageDog)
plt.subplot(3, 3, 9)
plt.imshow(imageDog, cmap='gray')
plt.show()
"""

# Подавление немаксимумов
suppression(image, dir1)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
"""
# 1
suppression(imageAngel, dir1)
plt.subplot(2, 3, 1)
plt.imshow(imageAngel, cmap='gray')
# 2
suppression(imageCat, dir2)
plt.subplot(2, 3, 2)
plt.imshow(imageCat, cmap='gray')
# 3
suppression(imageDog, dir3)
plt.subplot(2, 3, 3)
plt.imshow(imageDog, cmap='gray')
"""

# Гистерезис (уточнение границ)
image = clarification(image)
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
"""
# 1
imageAngel = clarification(imageAngel)
plt.subplot(2, 3, 4)
plt.imshow(imageAngel, cmap='gray')
# 2
imageCat = clarification(imageCat)
plt.subplot(2, 3, 5)
plt.imshow(imageCat, cmap='gray')
# 3
imageDog = clarification(imageDog)
plt.subplot(2, 3, 6)
plt.imshow(imageDog, cmap='gray')
"""

plt.show()

