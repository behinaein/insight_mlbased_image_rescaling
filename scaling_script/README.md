
To use this model to rescale an image, put your images that you need to scale in a folder and provid the path to model, images, and output path, and specify the paddig on the command line.

For instance, if you want to use this program to rescale images using a model named "mymodel.pth" in the "myimages" folder, with 50 pixel padding on all sides and put the outputs into "outputdir" with postfix "dnf_model" added to the resulted images, use the following command

```
python batch_image_rescaling -i /myimagefolder -m mymodel.pth -t 50 -b 50 -l 50 -r 50 -o /outputdir -d dnf_model
```
 
    
