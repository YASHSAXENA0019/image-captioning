from flask import Flask, render_template, request
import os
from backend import predict_step

app = Flask(__name__)

# Configure the path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Add route to handle image upload and caption generation
@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file from the request
    image_file = request.files['image']

    # Get the number of captions to generate from the request
    num_captions = int(request.form['num_captions'])

    # Save the uploaded image to the upload folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Generate captions for the uploaded image
    generated_captions = predict_step([image_path], num_captions=num_captions)

    # Remove the uploaded image after generating captions
    os.remove(image_path)

    # Render the result page with the generated captions
    return render_template('result.html', captions=generated_captions)


if __name__ == '__main__':
    app.run(debug=True)
