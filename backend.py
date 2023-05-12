from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

# Load the pre-trained model for image captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the feature extractor for the Vision Transformer (ViT) model
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the tokenizer for the ViT model
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Check if CUDA is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the maximum caption length and number of beams for beam search
max_length = 16
num_beams = 15

# Define the generation keyword arguments for the caption generation
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_paths, num_captions):
    images = []
    
    # Load and preprocess the input images
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
        
    # Extract the pixel values from the images and move them to the appropriate device
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generate captions for the input images
    output_ids = model.generate(pixel_values, **gen_kwargs, num_return_sequences=num_captions)

    # Decode the output caption IDs using the tokenizer
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Process and clean the generated captions
    captions = []
    for i in range(len(preds)):
        caption = preds[i]
        caption = caption.strip()
        captions.append(caption)

    return captions




