import numpy as np
import cv2
import random
import math


########## Masking Strategies ##########
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    """
    :param maxVertex: 数目
    :param maxLength: 长度
    :param maxBrushWidth: 宽度
    :param maxAngle:
    :param h:
    :param w:
    :return:
    """
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_stroke_mask(im_size, maxAngle=360, mask_scale=1):
    maxLength = im_size[0]
    maxVertex = im_size[0] // (70/mask_scale)
    maxBrushWidth = im_size[0] // (35/mask_scale)

    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(5, 13)
    # print(parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength,
                                        maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)
    return mask


def generate_square_mask(im_size, mask_scale=1):
    mask_size = im_size // (5 / mask_scale)
    mask_center = (np.random.randint(mask_size//2, im_size-mask_size//2), np.random.randint(mask_size//2, im_size-mask_size//2))
    min_x, max_x, min_y, max_y = \
        mask_center[0] - mask_size // 2, mask_center[0] + mask_size // 2, \
        mask_center[1] - mask_size // 2, mask_center[1] + mask_size // 2

    mask = np.zeros((1, 1, im_size, im_size))
    mask[:, :, int(min_x):int(max_x), int(min_y):int(max_y)] = 1
    return mask


if __name__ == "__main__":
    # generate image masks
    mask = generate_stroke_mask([352, 352])
    mask_square = generate_square_mask(352)

    mask = np.uint8(mask.squeeze()) * 255
    mask_square = np.uint8(mask_square.squeeze()) * 255
    cv2.imshow("mask", mask)
    cv2.waitKey(1000)

    cv2.imshow("mask", mask_square)
    cv2.waitKey(1000)

    pass