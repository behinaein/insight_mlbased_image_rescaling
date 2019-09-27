This program use a DFNet work to rescale images (change size and aspect ratio).

To use the model to rescale the model, please put your images that need to be scaled with the same padding in the a foder and provid the model path, images path, and paddig on the command line for the program

For instance, if you want to use this program to rescale images on the "myimages" folder, with 50 pixel padding on all sides using mymodel.pth and put the outputs into "outputdir" with "dnf_model" added to the resulted images, use the following command

python batch_image_rescaling -i /myimagefolder -m mymodel.pth -t 50 -b 50 -l 50 -r 50 -o /outputdir -d dnf_model

 
    
