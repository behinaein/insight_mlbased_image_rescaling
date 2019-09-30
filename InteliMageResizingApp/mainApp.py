from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory


# folder of images and the files allowed to be uploaded
UPLOAD_FOLDER = './static/img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create the application object
app = Flask(__name__)

# configure the upload folder and the maximum size of the files that can be uploaded
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    """
    only allows the file in the format specified by ALLOWED_EXTENSIONS to be used.
    :param filename: image file name
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# first page of the application
@app.route('/',methods=["GET","POST"])
def home_page():
    # initialize the images for the first load
    image_file_name = "screen.png"
    rescaled_file_name = "rescaled.png"

    # upload an image file
    # adopted from http://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_file_name = filename
            return render_template('index.html', _anchor='projects', input_image= os.path.join("static/img", image_file_name),
                                   )

    return render_template('index.html', input_image= os.path.join("static/img", image_file_name))  # render a template

# @app.route('/output')
# def tag_output():
# #       
#        # Pull input
#        some_input =request.args.get('user_input')            
       
#        # Case if empty
#        if some_input == '':
#            return render_template("index.html",
#                                   my_input = some_input,
#                                   my_form_result="Empty")
#        else:
#            some_output="yeay!"
#            some_number=3
#            some_image="giphy.gif"
#            return render_template("index.html",
#                               my_input=some_input,
#                               my_output=some_output,
#                               my_number=some_number,
#                               my_img_name=some_image,
#                               my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
    app.secret_key = 'hWNzK7zwgSboMNR4ZV)a4dx&#o9fo^2%DD*#w%Zf)C31V6&$xy' # key for uploading file securely
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

