import gradio as gr
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import os

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Inference function
def kosmos2_infer(image, prompt):
    image.save("temp_image.jpg")  # Normalize input like Kosmos-2 demo
    image = Image.open("temp_image.jpg")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    raw_output = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    clean_text, entities = processor.post_process_generation(generated_text)

    return clean_text, raw_output, str(entities)

# Interface
demo = gr.Interface(
    fn=kosmos2_infer,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(value="<grounding>An image of", label="Prompt"),
    ],
    outputs=[
        gr.Text(label="Cleaned Caption"),
        gr.Text(label="Raw Output with Tags"),
        gr.Text(label="Extracted Entities (Text, Positions, Bounding Boxes)"),
    ],
    title="Kosmos-2 Multimodal Grounding Demo",
    description="Upload an image and enter a grounding prompt like '<grounding>An image of a <dog> and a <cat></grounding>'.",
    examples=[
        [
            "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png",
            "<grounding>An image of a <snowman> warming himself by a <fire>.</grounding>"
        ]
    ]
)

demo.launch()
