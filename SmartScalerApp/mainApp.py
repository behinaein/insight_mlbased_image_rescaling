from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory
from flask import session
from utils_dfn import batch_image_rescaling

# folder of images and the files allowed to be uploaded
UPLOAD_FOLDER = './static/img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create the application object
app = Flask(__name__)

# configure the upload folder and the maximum size of the files that can be uploaded
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2 * 1024


def allowed_file(filename):
    """
    only allows the file in the format specified by ALLOWED_EXTENSIONS to be used.
    :param filename: image file name
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# first page of the application
@app.route('/', methods=["GET"])
@app.route('/index.html', methods=["GET"])
def home_page():
    """
    Show the front page of the website
    adopted from http://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
    :return: the main page: index.html
    """
    return render_template('index.html',
                           input_image=os.path.join("static/img", session['input_image_name']),
                           output_image=os.path.join("static/img", session['rescaled_file_name']))  # render a template


@app.route('/rescaling.html', methods=["GET", "POST"])
def rescale_image():
    """
    show the rescaling page
    :return: rescaling page
    """
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
            # upload the file in the folder specified by "UPLOAD_FOLDER"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['input_image_name'] = filename
            # return the rescaling.html with input image uploaded
            redirect(url_for('rescale_image', _anchor='projects'))  #

    return render_template('rescaling.html',
                           input_image=os.path.join("static/img", session['input_image_name']),
                           output_image=os.path.join("static/img", session['rescaled_file_name']))


@app.route('/convert_image', methods=["Post"])
def convert_image():
    """
    Runs the DNF model and return rescale_img.html with the rescaled image
    :return: rescaling.html
    """
    if request.path == "/convert_image":
        # read the values for the height and width
        new_height = request.form.get('height_size_input')
        new_width = request.form.get('width_size_input')
        # initialized the rescaling class
        batch_processor = batch_image_rescaling.BatchImageRescaling()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        input_file_name = os.path.join(dir_path, 'static/img', session['input_image_name'])
        try:
            # check if the new_height and new_width are OK
            new_height = int(new_height)
            new_width = int(new_width)
        except Exception:
            return render_template('rescaling.html',
                                   input_image=os.path.join("static/img", session['input_image_name']),
                                   output_image=os.path.join("static/img", session['rescaled_file_name']))
        # run the rescaling model
        batch_processor.rescale_image(input_file_name, new_width, new_height,
                                      'transfer_learning_26_09_2019-10_30_54.pt', '_new_prog')
        session['rescaled_file_name'] = batch_processor.get_rescaled_file_name()
        return render_template('rescaling.html',
                               _anchor='projects',
                               input_image=os.path.join("static/img", session['input_image_name']),
                               output_image=os.path.join("static/rescaled_images", session['rescaled_file_name']))


@app.before_first_request
def startup():
    """
    initialize parameters
    :return: None
    """
    session['input_image_name'] = "Input_image_template.png"
    session['rescaled_file_name'] = "rescaled_image_template.png"
    session['webpage'] = ""
    session['anchor_to_go'] = ""
    session['new_height'] = "new_height"
    session['new_weight'] = "new_weight"


# start the server with the 'run()' method
if __name__ == "__main__":
    app.secret_key = 'hWNzK7zwgSboMNR4ZV)a4dx&#o9fo^2%DD*#w%Zf)C31V6&$xy'  # key for uploading file securely
    app.run(debug=True)  # will run locally http://127.0.0.1:5000/
