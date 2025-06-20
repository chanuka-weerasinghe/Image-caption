# 🖼️ AI Image Caption Generator

A beautiful web application that generates descriptive captions for images using the BLIP image captioning model.

## 📋 Features

- Upload images to generate AI-powered captions
- Beautiful, modern UI with responsive design
- Example images to try out the application
- Automatic saving of uploaded images
- Easy to use interface

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-image-caption.git
cd ai-image-caption
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the application:
```bash
python caption_generator.py
```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860)

3. Upload an image or use one of the provided examples

4. Click the "Generate Caption" button to get an AI-generated description of your image

## 🔧 Technologies Used

- [BLIP](https://github.com/salesforce/BLIP) - Image captioning model
- [Gradio](https://gradio.app/) - Web interface framework
- [Transformers](https://huggingface.co/docs/transformers/index) - Hugging Face's library for state-of-the-art ML models
- [PIL](https://python-pillow.org/) - Python Imaging Library

## 📁 Project Structure

- `caption_generator.py` - Main application file
- `uploads/` - Directory where uploaded images are saved
- `requirements.txt` - List of dependencies

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the model
- [Unsplash](https://unsplash.com/) for the example images
- [Icons8](https://icons8.com/) for the header icon 