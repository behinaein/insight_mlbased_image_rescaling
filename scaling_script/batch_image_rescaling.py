import os
import shutil
from pad_image_and_create_mask import mask_pad_generator
import argparse
from dfnet.test import Tester


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
        self.file_name_with_ext = input_file.split('/')[-1]
        file_name = self.file_name_with_ext.split('.')[:-1]
        self.file_name = ''.join(file_name)

    def run_padding(self, top, bottom, left, right):
        """
        adds padding to the side of images, create a mask, and moves the results to the imag and mask folders
        :param top: number of pixels to be added to the top
        :param bottom: number of pixels to be added to the bottom
        :param left: number of pixels to be added to the left
        :param right: number of pixels to be added to the right
        :return: None
        """
        mask_pad_generator(os.path.join('./temp', self.file_name_with_ext), top, bottom, left, right, './temp')
        shutil.move(os.path.join('./temp', self.file_name + '_padded_resized.png'), 'img')
        shutil.move(os.path.join('./temp', self.file_name + '_mask.png'), 'mask')

    def run_dfn(self, model, file_description):
        """
        runs the DNF model on the image and stores the results in the compare folder
        :param model: DNF model to use
        :param file_description: the description to be add the  resulted images
        :return:
        """
        dfn_folder = './dfnet'
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        model = os.path.join(dfn_folder, 'model/' + model)
        image_dir = os.path.join(curr_dir, 'img')
        mask_dir = os.path.join(curr_dir, 'mask')
        output_dir = os.path.join(curr_dir, 'output')

        run_model = Tester(model, 0, 8)  # create an instance of the Tester class of the model to run the model
        run_model.inpaint(output_dir, image_dir, mask_dir, merge_result=False)
        file_name = os.listdir('./output/result')
        new_file_name = self.file_name + file_description + '.png'
        shutil.move(os.path.join('./output/result', file_name[0]), os.path.join('./compare', new_file_name))


def create_required_dir():
    """
    creates the directories for storing original, intermediate, and end results
    :return: None
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    if not os.path.exists('img'):
        os.mkdir('img')
    if not os.path.exists('mask'):
        os.mkdir('mask')
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('compare'):
        os.mkdir('compare')


def clean():
    """
    removes any files and subfolders that temporary created in the rescaling process
    :return: None
    """
    folders = ['./temp', './img', './mask', './output']
    for folder in folders:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)


if __name__ == '__main__':
    # read the required parameter from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image_dir', default='/home/beh/Desktop/temp/images_rescaled/sample2/original.png',
        help='Select an input image dir')
    parser.add_argument(
        '-m', '--model', default="model_places2.pth",
        help='DNF model.')
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
    parser.add_argument(
        '-d', '--description', default='dfn_org',
        help='postfix that describes the image'
    )

    args = parser.parse_args()
    batch_processor = BatchImageRescaling()
    file_list = os.listdir(args.image_dir)

    create_required_dir()
    for file in file_list:
        input_file_name = os.path.join(args.image_dir, file)
        shutil.copy(input_file_name, "./temp")
        batch_processor.extract_file_name(input_file_name)
        batch_processor.run_padding(args.top, args.bottom, args.left, args.right)
        batch_processor.run_dfn(args.model, args.description)
        clean()
