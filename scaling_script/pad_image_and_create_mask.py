import cv2
import argparse 
import numpy as np
import os


def mask_pad_generator(image_file, top, bottom, left, right, output_folder=None):
    """
    add pads to the image, rescale the image to be suitable size for DNF, and create the mask
    :param image_file: image file path
    :param top: number of pixels to be added to the top
    :param bottom: number of pixels to be added to the bottom
    :param left: number of pixels to be added to the left
    :param right: number of pixels to be added to the right
    :param output_folder: the name of the folder to store the padded image and mask
    :return: paddes_image, mask
    """
    
    input_image = cv2.imread(image_file)
    value = [0, 0, 0]   # add black padding
    height = input_image.shape[0]
    width = input_image.shape[1]
    new_height = height + top + bottom
    new_width = width + right + left 
    diff = new_height - new_width
    # add extra pad to make the image square so it is not distorted in resizing
    if diff > 0:
        # extra_padding_right = diff//2 
        # extra_padding_left = diff//2 if diff%2 == 0 else diff//2 + 1
        # right += extra_padding_right
        # left += extra_padding_left
        new_width += diff
        right += diff
    elif diff < 0:
        diff = -diff
        # extra_padding_top = diff//2 
        # extra_padding_bottom = diff//2 if diff%2 == 0 else diff//2 + 1
        # top += extra_padding_top
        # bottom += extra_padding_bottom
        new_height += diff
        bottom += diff

    # create a white mask and then add black padding
    mask = 255 * np.ones((new_height, new_width), dtype=np.uint8)
    mask[:top, :] = 0
    mask[new_height - bottom:, :] = 0
    mask[:, :left] = 0
    mask[:, new_width - right:] = 0
    mask = mask[..., np.newaxis]
    image_padded = cv2.copyMakeBorder(input_image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
    image_padded_resized = cv2.resize(image_padded, (512, 512))
    mask = cv2.resize(mask, (512, 512))

    if output_folder is not None:
        file_name_with_extension = image_file.split('/')[-1]
        file_name = file_name_with_extension.split('.')[:-1]
        file_name = ''.join(file_name)
        cv2.imwrite(os.path.join(output_folder, file_name + '_padded_resized.png'), image_padded_resized)
        cv2.imwrite(os.path.join(output_folder, file_name + '_mask.png'), mask)

    return image_padded, mask


if __name__ == '__main__':
    # read the required arguments from input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image', default='/home/beh/Desktop/temp/images_rescaled/sample2/original.png',
        help='Select an input image.')
    parser.add_argument(
        '-t', '--top', default=0, type=int,
        help='top pad size.')
    parser.add_argument(
        '-b', '--bottom', default=0, type=int,
        help='bottom pad size.')
    parser.add_argument(
        '-l', '--left', default=0, type=int,
        help='left pad size.')
    parser.add_argument(
        '-r', '--right', default=0, type=int,
        help='right pad size.')
    parser.add_argument(
        '-o', '--output', default='./output/',
        help='Output dir')

    args = parser.parse_args()

    mask_pad_generator(args.image, args.top, args.bottom, args.left, args.right, args.output)
