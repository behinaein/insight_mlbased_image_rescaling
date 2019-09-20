import numpy as np
import scipy
from numba import jit
import cv2
from scipy.ndimage import convolve1d

"""
Code adopted from https://github.com/andrewdcampbell/seam-carving/blob/master/seam_carving.py#L242
This program performs image resizing using a version of seam-carving
"""

class SeamCarver:

    def __init__(self, img, new_width, new_height, preserve_mask=None):
        """
        initialize the object
        :param img: the original image
        :param new_width: the new width for the image (x)
        :param new_height: the new height for the image (y)
        :param preserve_mask: a boolean matrix the same size as image that indicates where should be preserved: True preserved
        """
        self.img = img
        self.new_width = new_width
        self.new_height = new_height
        self.preserve_mask = preserve_mask
        self.ENERGY_FOR_PRESERVED_AREA = 1e6

    @jit
    def compute_energy(self, img:np.ndarray):
        """
        computes energy for the img to be used for seleting the minimal seams
        :return: energy matrix
        """

        # computing the energy by calculating gradient along x and y axes and use the square
        # root of the sum of each gradient squared
        xgrad = convolve1d(img, np.array([1, 0, -1]), axis=1, mode='wrap')
        ygrad = convolve1d(img, np.array([1, 0, -1]), axis=0, mode='wrap')

        energy = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

        # mask contains the part of the image should be protected from altering
        if self.preserve_mask is not None:
            energy[self.preserve_mask] = self.ENERGY_FOR_PRESERVED_AREA

        return energy

    def resize(self, width, height):
        """
        resizes image to new width and height
        :param width: width of the new image
        :param height: height of the new image
        :return: resized image
        """
        return cv2.resize(self.img, (width, height))

    def rotate_image(self, img, clockwise):
        """
        rotates the image 90 degrees clockwise or anticlockwise
        :param img: the image to be rotated
        :param clockwise: True for clockwise rotation and False for anticlockwise rotation
        :return: rotated image
        """
        k = 1 if clockwise else 3
        return np.rot90(img, k)

    @jit
    def minimum_seam(self, img):
        """
        finds and return the miniaml path (seam) from each row and the pixels that must be removed
        :param img: the image that one would like to find the seams for
        :return: two arrays, one containing the indices of the pixels in each seams, and a logical array
        shows which pixels must be removed
        """
        height, width, _ = img.shape

        energy = self.compute_energy(img)
        backtrack = np.zeros_like(energy, dtype=np.int)

        for i in range(1, height):
            for j in range(0, width):
                # Handle the left edge of the image, to ensure we don't index a -1
                if j == 0:
                    idx = np.argmin(energy[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = energy[i - 1, idx + j]
                else:
                    idx = np.argmin(energy[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = energy[i - 1, idx + j - 1]

                energy[i, j] += min_energy

        seam_idx = []
        mask_boolean = np.ones((height, width), dtype=np.bool)
        j = np.argmin(energy[-1])
        for i in range(height - 1, -1, -1):
            mask_boolean[i, j] = False
            seam_idx.append(j)
            j = backtrack[i, j]

        seam_idx.reverse()
        return np.array(seam_idx), mask_boolean

    @jit
    def remove_seam(self, img, mask_boolean):
        """
        removes pixels using mask_boolean
        :param img: input image that we would like to shrink
        :param mask_boolean: a boolean array that show which pixels must be removed by marking them as False
        :return: shrank image
        """
        if len(img.shape) == 2:
            height, width = img.shape
            return img[mask_boolean].reshape((height, width - 1))
        elif len(img.shape) == 3:
            height, width, _ = img.shape
            mask3c_booelan = np.stack([mask_boolean] * 3, axis=2)
            return img[mask3c_booelan].reshape((height, width - 1, 3))

    @jit
    def add_seam(self, img, seam_idx):
        """
        adds a seam to an image
        :param img: the image that need to be enlarged
        :param seam_idx: index of the pixels that an average of their neighbors will be inserted in their place
        :return: the enlarged image
        """

        # check if the image or boolean mask, since the mask is a 2D array and image is
        # a 3D array
        if len(img.shape) == 2:  # boolean mask
            num_channels = 1
        elif len(img.shape) == 3: # image
            num_channels = img.shape[2]
        else:
            raise Exception('Wrong number of dimension')

        if num_channels == 1:  # this part is for the boolean mask
            height, width = img.shape
            out_image = np.zeros((height, width + 1), dtype=np.bool)
            for row in range(height):
                col = seam_idx[row]
                if col == 0:    # if this is the first column use the next two columns for averaging
                    p = np.average(img[row, col: col + 2])
                    out_image[row, col] = img[row, col]
                    out_image[row, col + 1] = p
                    out_image[row, col + 1:] = img[row, col:]
                else:   # if this is not  the first column use the previous and next columns for averaging
                    p = np.average(img[row, col - 1: col + 1])
                    out_image[row, : col] = img[row, : col]
                    out_image[row, col] = p
                    out_image[row, col + 1:] = img[row, col:]
        elif num_channels == 3:     # this part is for the image
            height, width, _ = img.shape
            out_image = np.zeros((height, width + 1, num_channels))
            for row in range(height):
                col = seam_idx[row]
                for ch in range(num_channels):
                    if col == 0:    # if this is the first column use the next two columns for averaging
                        p = np.average(img[row, col: col + 2, ch])
                        out_image[row, col, ch] = img[row, col, ch]
                        out_image[row, col + 1, ch] = p
                        out_image[row, col + 1:, ch] = img[row, col:, ch]
                    else:       # if this is not  the first column use the previous and next columns for averaging
                        p = np.average(img[row, col - 1: col + 1, ch])
                        out_image[row, : col, ch] = img[row, : col, ch]
                        out_image[row, col, ch] = p
                        out_image[row, col + 1:, ch] = img[row, col:, ch]

        return out_image

    @jit
    def seams_removal(self, img,  num_remove):
        """
        remove num_remove seams from an image
        :param img: input image
        :param num_remove: number of seams that need to be removed from the image
        :return: shrank image
        """
        for i in range(num_remove):
            seam_idx, mask_boolean = self.minimum_seam(img)
            im = self.remove_seam(img, mask_boolean)
            if self.preserve_mask is not None:
                self.preserve_mask = self.remove_seam(self.preserve_mask, mask_boolean)

        return im

    @jit
    def seams_insertion(self, img, num_add):
        """
        adds num_add seams to an image
        :param img: input image
        :param num_add: number of seams to be added to the image
        :return: enlarged image
        """
        for i in range(num_add):
            seam_idx, mask_boolean = self.minimum_seam(img)
            img = self.add_seam(img, seam_idx)
            if self.preserve_mask is not None:
                self.preserve_mask = self.add_seam(self.preserve_mask, seam_idx)

        return img


    def seam_carve(self):
        """
        This function decides to shrink or enlarge the data according tho the new sizes provided and return the image
        :return: the alterd image
        """
        img = self.img.astype(np.float64)
        dx = self.new_width - img.shape[1]
        dy = self.new_height - img.shape[0]

        height, width, _ = img.shape
        assert height + dy > 0 and width + dx > 0 and dy <= height and dx <= width

        output = img

        if dx < 0: #image need to be shrank along the colomns (width)
            output = self.seams_removal(output, -dx)

        elif dx > 0: #image need to be enlarged along the colomns (width)
            output = self.seams_insertion(output, dx)

        # since we have written the shrinking and enlarging functions only for width,
        # to change the height we need to rotate the input image

        if dy < 0: #image need to be shrank along the rows (height)
            output = self.rotate_image(output, True)
            if self.preserve_mask is not None:
                self.preserve_mask = self.rotate_image(self.preserve_mask, True)
            output = self.seams_removal(output, -dy)
            output = self.rotate_image(output, False)
            if self.preserve_mask is not None:
                self.preserve_mask = self.rotate_image(self.preserve_mask, True)

        elif dy > 0:  #image need to be enlarged along the rows (height)
            output = self.rotate_image(output, True)
            if self.preserve_mask is not None:
                self.preserve_mask = self.rotate_image(self.preserve_mask, True)
            output = self.seams_insertion(output, dy)
            output = self.rotate_image(output, False)
            if self.preserve_mask is not None:
                self.preserve_mask = self.rotate_image(self.preserve_mask, True)
        return output

if __name__ == '__main__':
    image_file = '/home/beh/Downloads/original.png'
    img = cv2.imread(image_file)
    print(img.shape)

    preserve_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    preserve_mask[100:800,:1080] = np.True_
    seam_carving = SeamCarver(img, 1080, 1080, preserve_mask)
    new_image = seam_carving.seam_carve()
    print(new_image.shape)
    cv2.imwrite('/home/beh/Downloads/orgin1.png' , new_image.astype(np.uint8))

