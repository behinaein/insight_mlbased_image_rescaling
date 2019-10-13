import os
import shutil
from utils_dfn.pad_image_and_create_mask import gen_padded_image_and_mask, crop_image, resize_image, extract_file_name
from utils_dfn.dfnet.run_model import RunModel
import cv2


class BatchImageRescaling:
    """
    Class to do bach rescaling on images
    """

    def __init__(self):
        self.file_name = None
        self.file_name_with_ext = None
        create_required_dir()

    def extract_file_name(self, input_file):
        """
        extracts file name with and without extension from the full file path
        :param input_file: full file path
        :return: None
        """
        self.file_name_with_ext, self.file_name  = extract_file_name(input_file)


    def run_padding(self):
        """
        adds padding to the side of images, create a mask, and moves the results to the imag and mask folders
        :return: None
        """

        image_padded, mask, self.pad_to_right, self.pad_to_bottom = gen_padded_image_and_mask (os.path.join('utils_dfn/temp', self.file_name_with_ext),
                                                                        self.new_height, self.new_width)
        cv2.imwrite(os.path.join('utils_dfn/img', self.file_name + '_padded_resized.png'), image_padded)
        cv2.imwrite(os.path.join('utils_dfn/mask', self.file_name + '_mask.png'), mask)

    def run_dfn(self, model):
        """
        runs the DNF model on the image and stores the results in the compare folder
        :param model: DNF model to use
        :param file_description: the description to be add the  resulted images
        :return:
        """
        dfnet_folder = 'utils_dfn/dfnet'
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        model = os.path.join(dfnet_folder, 'model',  model)
        image_dir = os.path.join(curr_dir, 'img')
        mask_dir = os.path.join(curr_dir, 'mask')
        output_dir = os.path.join(curr_dir, 'output')

        run_model = RunModel(model, 0, 8)  # create an instance of the RunModel class of the model to run the model
        run_model.inpaint(output_dir, image_dir, mask_dir, merge_result=False)


    def restore_to_correct_size(self, file_description):
        """
        changes the ouput of the model to correct size by removing extra padding and resizing
        :param file_description: post_fix of the file name
        :return: None
        """
        file_name = os.listdir('utils_dfn/output/result')[0]
        file_name_with_ext, file_name = extract_file_name(file_name)
        new_file = os.path.join('static/rescaled_images', self.file_name + file_description + '.png')
        img = cv2.imread(os.path.join('utils_dfn/output/result', file_name_with_ext))
        if self.new_width > self.new_height:
            new_size = self.new_width
        else:
            new_size = self.new_height

        img = resize_image(img, new_size, new_size)
        img = crop_image(img, bottom=self.pad_to_bottom, right=self.pad_to_right)
        cv2.imwrite(new_file, img)
        self.rescaled_file_name_with_ext, _ = extract_file_name(new_file)

    def get_rescaled_file_name(self):
        return self.rescaled_file_name_with_ext

    def rescale_image(self, img_file, new_width, new_height, model_path, file_description):
        """
        rescale an images to the new_width and new_height using the model provided and adds file_description
        to the result image
        :param img_file: input image file full path
        :param new_width: new desired width
        :param new_height: new desired height
        :param model_path: the path to the model to be used
        :param file_description: post_fix string to be added to the output file name
        :return: None
        """
        cwd = os.getcwd()
        self.new_width = new_width
        self.new_height = new_height
        self.extract_file_name(img_file)
        shutil.copy(img_file, os.path.join('utils_dfn/temp', self.file_name_with_ext))
        self.run_padding()
        self.run_dfn(model_path)
        self.restore_to_correct_size(file_description)
        clean()

def create_required_dir():
    """
    creates the directories for storing original, intermediate, and end results
    :return: None
    """
    if not os.path.exists('utils_dfn/temp'):
        os.mkdir('utils_dfn/temp')
    if not os.path.exists('utils_dfn/img'):
        os.mkdir('utils_dfn/img')
    if not os.path.exists('utils_dfn/mask'):
        os.mkdir('utils_dfn/mask')
    if not os.path.exists('utils_dfn/output'):
        os.mkdir('utils_dfn/utils_dfn/output')
    # if not os.path.exists('compare'):
    #     os.mkdir('compare')


def clean():
    """
    removes any files and subfolders that temporary created in the rescaling process
    :return: None
    """
    folders = ['utils_dfn/temp', 'utils_dfn/img', 'utils_dfn/mask', 'utils_dfn/output']
    for folder in folders:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)


if __name__ == '__main__':
    pass
    # read the required parameter from the command line
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-i', '--image_dir', default='/home/beh/Desktop/temp/images_rescaled/sample2/original.png',
    #     help='Select an input image dir')
    # parser.add_argument(
    #     '-m', '--model', default="model_places2.pth",
    #     help='DNF model.')
    # parser.add_argument(
    #     '-t', '--top', default=0, type=int,
    #     help='top pad size.')
    # parser.add_argument(
    #     '-b', '--bottom', default=0, type=int,
    #     help='bottom pad size.')
    # parser.add_argument(
    #     '-l', '--left', default=0, type=int,
    #     help='left pad size.')
    # parser.add_argument(
    #     '-r', '--right', default=0, type=int,
    #     help='right pad size.')
    # parser.add_argument(
    #     '-o', '--output', default='./output/',
    #     help='Output dir')
    # parser.add_argument(
    #     '-d', '--description', default='dfn_org',
    #     help='postfix that describes the image'
    # )
    #
    # args = parser.parse_args()
    # batch_processor = BatchImageRescaling()
    # file_list = os.listdir(args.image_dir)
    #
    # create_required_dir()
    # for file in file_list:
    #     input_file_name = os.path.join(args.image_dir, file)
    #     shutil.copy(input_file_name, "./temp")
    #     batch_processor.extract_file_name(input_file_name)
    #     batch_processor.run_padding(args.top, args.bottom, args.left, args.right)
    #     batch_processor.run_dfn(args.model, args.description)
    #     clean()
    # batch_processor = BatchImageRescaling()
    # input_file_name = '/home/beh/Documents/insight_project/datasets/original_images/plate.png'
    # batch_processor.rescale_image(input_file_name, 1500, 750,
    #                               'transfer_learning_26_09_2019-10_30_54.pt', '_new_prog')
