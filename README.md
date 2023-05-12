# Image Captioning Web Application

This project is a Flask-based web application that utilizes a pre-trained image captioning model to generate captions for uploaded images. The application allows users to upload an image and receive multiple captions describing the content of the image.

## Installation

1. Clone the repository to your local machine:

```
git clone https://github.com/your-username/image-captioning-web-app.git
```

2. Install the required dependencies. It is recommended to use a virtual environment for this step:

```
cd image-captioning-web-app
pip install -r requirements.txt
```

3. Download the pre-trained image captioning model and tokenizer:

```python
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# Load the pre-trained model for image captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the feature extractor for the Vision Transformer (ViT) model
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the tokenizer for the ViT model
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Check if CUDA is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## Usage

1. Start the Flask web server:

```
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Upload an image using the provided form.

4. Click the "Generate Captions" button.

5. Wait for the captions to be generated.

6. View the generated captions on the result page.

## Project Structure

The project consists of the following files:

- `app.py`: This file contains the Flask application code. It defines routes for the home page and the image upload/caption generation functionality.
- `backend.py`: This file contains the backend code for the image caption generation. It loads the pre-trained model, defines the caption generation process, and provides the `predict_step` function used by the Flask application.
- `templates/index.html`: This HTML template defines the structure of the home page.
- `templates/result.html`: This HTML template defines the structure of the result page, where the generated captions are displayed.
- `uploads/`: This folder is the destination for uploaded images. The images are temporarily stored here before being processed and removed.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
