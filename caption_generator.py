from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
from gradio.themes import Soft
import requests
import os
from datetime import datetime

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    if image is None:
        return "Please upload an image to generate a caption."
    
    # Save the uploaded image with timestamp
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("uploads", f"image_{timestamp}.jpg")
    if image is not None:
        image.save(image_path)
    
    # Generate caption
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_length=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption

# Example images for the demo
example_images = [
    "https://images.unsplash.com/photo-1575936123452-b67c3203c357?q=80&w=1000",
    "https://images.unsplash.com/photo-1618588507085-c79565432917?q=80&w=1000",
    "https://images.unsplash.com/photo-1682687220063-4742bd7fd538?q=80&w=1000"
]

# Create a custom theme
custom_theme = Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="gray",
)

# CSS for custom styling
custom_css = """
.container {
    max-width: 1000px;
    margin: auto;
}
.gradio-container {
    font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.caption-output {
    font-size: 1.2rem;
    padding: 1.2rem;
    border-radius: 0.5rem;
    background: #f0f7fa;
    border-left: 4px solid #009688;
    margin-top: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    font-size: 0.9rem;
    color: #666;
    border-top: 1px solid #eee;
}
h1 {
    color: #009688;
    text-align: center;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #607D8B;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}
.generate-btn {
    background: linear-gradient(90deg, #009688, #4DB6AC);
}
.header-img {
    display: block;
    margin: 0 auto 1rem auto;
    max-width: 100px;
}
.examples-header {
    font-weight: 600;
    margin-top: 2rem;
    color: #607D8B;
    text-align: center;
}
"""

# Gradio interface
with gr.Blocks(css=custom_css, theme=custom_theme) as interface:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
            <img src="https://img.icons8.com/fluency/96/000000/image.png" class="header-img" alt="Image Caption Logo">
            <h1>🖼️ AI Image Caption Generator</h1>
            <p class="subtitle">Transform your images into descriptive captions using AI</p>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil", 
                    label="Upload Your Image",
                    elem_id="image-input",
                    height=300
                )
                generate_btn = gr.Button(
                    "✨ Generate Caption", 
                    variant="primary",
                    elem_classes="generate-btn"
                )
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Generated Caption", 
                    placeholder="Your caption will appear here...",
                    elem_classes="caption-output",
                    lines=5
                )
        
        gr.HTML("<h3 class='examples-header'>Try with these examples:</h3>")
        
        gr.Examples(
            examples=example_images,
            inputs=input_image,
            outputs=output_text,
            fn=generate_caption,
            cache_examples=True,
        )
        
        gr.HTML("""
            <footer>
                <p>Powered by BLIP Image Captioning Model | Created with Gradio</p>
                <p>© 2023 AI Image Caption Generator</p>
            </footer>
        """)

if __name__ == "__main__":
    interface.launch()
