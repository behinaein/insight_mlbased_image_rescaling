import cv2
import argparse 
import numpy as np
import os


def add_pads(img, file_name, top, bottom, left, right, output_folder=None):
    """
    add pads to the image and rescale the image to be suitable size for DNF. If output_folder is given, mask
    will be saved in that folder.
    :param img: an image
    :param file_name: the name fo the image file
    :param top: number of pixels to be added to the top
    :param bottom: number of pixels to be added to the bottom
    :param left: number of pixels to be added to the left
    :param right: number of pixels to be added to the right
    :param output_folder: the name of the folder to store the padded image and mask
    :return: padded_image, top, bottom, left, and right
    """
    value = [0, 0, 0]   # add black padding
    height = img.shape[0]
    width = img.shape[1]
    new_height = height + top + bottom
    new_width = width + right + left 
    diff = new_height - new_width
    pad_to_right = 0
    pad_to_bottom = 0
    # add extra pad to make the image square so it is not distorted in resizing
    if diff > 0:
        # extra_padding_right = diff//2
        # extra_padding_left = diff//2 if diff%2 == 0 else diff//2 + 1
        # right += extra_padding_right
        # left += extra_padding_left
        new_width += diff
        right += diff
        pad_to_right = diff

    elif diff < 0:
        diff = -diff
        # extra_padding_top = diff//2 
        # extra_padding_bottom = diff//2 if diff%2 == 0 else diff//2 + 1
        # top += extra_padding_top
        # bottom += extra_padding_bottom
        new_height += diff
        bottom += diff
        pad_to_bottom = diff

    # add black padding and resize to the CNN input size
    image_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
    image_padded_resized = cv2.resize(image_padded, (512, 512))

    if output_folder is not None:
        cv2.imwrite(os.path.join(output_folder, file_name + '_padded_resized.png'), image_padded_resized)

    return image_padded, top, bottom, left, right, pad_to_right, pad_to_bottom


def generate_mask(file_name, height, width, top, bottom, left, right, output_folder=None):
    """
    generates a mask (numpy arrays of 0 and 1) in the size of height and width, where everywhere is
    1 except the parts specified by top, bottom, left, and right. If output_folder is given, mask
    will be saved in that folder.
    :param file_name: the name of the image file
    :param height: height of the mask
    :param width: width of the mask
    :param top: amount of black margin on the top
    :param bottom: amount of black margin on the bottom
    :param left: amount of black margin on the left
    :param right: amount of black margin on the right
    :param output_folder: where to save the mask file
    :return: mask (a numpy array)
    """
    height += top + bottom
    width += left + right
    mask = 255 * np.ones((height, width), dtype=np.uint8)
    mask[:top+10, :] = 0
    mask[height - bottom-10:, :] = 0
    mask[:, :left+10] = 0
    mask[:, width - right-10:] = 0
    mask = mask[..., np.newaxis]
    mask = cv2.resize(mask, (512, 512))
    if output_folder is not None:
        cv2.imwrite(os.path.join(output_folder, file_name + '_mask.png'), mask)
    return mask


def resize_to_smallest_ratio(image_file, new_height, new_width):
    """
    resize a image to a fit into a frame where the size of frame is chosen to be  the smallest
    resizing needed fit the image along one of the
    height and width
    :param image_file: input image file
    :param new_height: new height
    :param new_width: new width
    :return: resized image
    """
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    height_ratio = new_height / height
    width_ratio = new_width / width
    convert_ratio = width_ratio if width_ratio < height_ratio else height_ratio
    height = int(height * convert_ratio)
    width = int(width * convert_ratio)
    resized_image = cv2.resize(img, (width, height),
                               interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('/home/beh/Desktop/temp/plate_resized.png', resized_image)
    return resized_image


def resize_image(img, new_height, new_width, file_name=None):
    """
    resize an image and saves the image if file_name is provided
    :param img: image to resize
    :param new_height: new height
    :param new_width: new width
    :param file_name: the file name with full path
    :return: the resized image
    """
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if file_name is not None:
        cv2.imwrite(file_name, resized_img)
    return resized_img


def extract_file_name(input_file):
    """
    extracts file name with and without extension from the full file path
    :param input_file: full file path
    :return: file_name with extension and file name without an extension
    """
    file_name_with_ext = input_file.split('/')[-1]
    file_name = file_name_with_ext.split('.')[:-1]
    file_name = ''.join(file_name)
    return file_name_with_ext, file_name

def compute_padding(img, new_height, new_width):
    """
    compute the amount of padding to be added to the top, bottom, left, and right of image
    to change it to the correct size
    :param img: input image
    :param new_height: new height
    :param new_width: new width
    :return: amount of padding needed on the top, bottom, left, and right
    """

    top = 0
    bottom = 0
    left = 0
    right = 0

    height, width, _ = img.shape

    if height < new_height:
        diff_h = new_height - height
        top = diff_h // 2
        bottom = diff_h//2 if diff_h % 2 == 0 else diff_h//2 + 1

    if width < new_width:
        diff_w = new_width - width
        left = diff_w // 2
        right = diff_w//2 if diff_w % 2 == 0 else diff_w//2 + 1

    return top, bottom, left, right


def gen_padded_image_and_mask(image_file_path, new_height, new_width, output_folder=None):

    _, file_name = extract_file_name(image_file_path)
    img = resize_to_smallest_ratio(image_file_path, new_height, new_width)
    top, bottom, left, right = compute_padding(img, new_height, new_width)
    image_padded, top, bottom, left, right, pad_to_right, pad_to_bottom= add_pads(img, file_name, top,
                                                      bottom, left, right,
                                                      output_folder)
    height, width, _ = img.shape
    mask = generate_mask(file_name, height, width, top, bottom,
                         left, right, output_folder)
    return image_padded, mask, pad_to_right, pad_to_bottom


def crop_image(img, top=0, bottom=0, left=0, right=0, file_name=None):
    """
    crops an image according to the margins provided and save it if output_folder is provided
    :param img: the image to be cropped
    :param top: number of pixels from top to remove
    :param bottom: number of pixels from bottom to remove
    :param left: number of pixels from left to remove
    :param right: number of pixels from right to remove
    :param file_name: the name of the file that the result will be saved in
    :return: cropped image
    """
    height, width, _ = img.shape
    temp_img = np.copy(img)
    cropped_image = temp_img[top:height-bottom, left:width-right]
    if file_name is not None:
        cv2.imwrite(file_name, cropped_image)
    return cropped_image


if __name__ == '__main__':
    # read the required arguments from input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image', default='/home/beh/Desktop/temp/images_rescaled/sample2/original.png',
        help='Select an input image.')
    parser.add_argument(
        '-t', '--height', default=512, type=int,
        help='New height.')
    parser.add_argument(
        '-w', '--width', default=512, type=int,
        help='New width.')
    parser.add_argument(
        '-o', '--output', default='./output/',
        help='Output dir.')

    args = parser.parse_args()
    gen_padded_image_and_mask(args.image, args.height, args.width, args.output)
