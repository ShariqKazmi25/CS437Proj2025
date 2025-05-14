
import torch
import argparse
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with YOLOv8s and CLIP on Trash in Water Channels dataset.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained YOLOv8 model checkpoint")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image or directory of images")
    parser.add_argument('--output', type=str, required=True, help="Directory to save inference results")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on (cpu/cuda)")
    return parser.parse_args()

# Initialize YOLO and CLIP models
def initialize_models(device):
    model = YOLO("yolov8s.pt").to(device)  # Load the YOLOv8 model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return model, clip_processor, clip_model

# Function to apply CLIP rescoring on detected bounding boxes
def clip_rescore(predictions, clip_processor, clip_model, image_path, device):
    pil_img = Image.open(image_path).convert("RGB")
    preds = []
    for box, score, class_id in zip(predictions.boxes.xyxy, predictions.boxes.conf, predictions.boxes.cls):
        # Crop the region of interest
        x1, y1, x2, y2 = box
        crop = pil_img.crop((x1, y1, x2, y2))
        
        # Process the cropped image with CLIP
        inputs = clip_processor(images=crop, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ text_embeds.T).squeeze(0).cpu().numpy()  # Compare with pre-computed text embeddings
        
        best_class = int(sims.argmax())  # Get best matching class
        fused_score = score * (1 - 0.5) + sims[best_class] * 0.5  # Rescore using fusion
        
        preds.append({
            "bbox": [x1, y1, x2, y2],
            "score": fused_score,
            "class_id": best_class
        })
    return preds

# Function to run inference and visualize results
def run_inference(model, clip_processor, clip_model, image_path, output_dir, device):
    # Read the image
    image = Path(image_path)
    img = Image.open(image).convert("RGB")
    
    # Run inference with YOLOv8
    results = model(image_path)
    predictions = results[0]  # Get the first prediction

    # Apply CLIP rescoring
    pred_boxes = clip_rescore(predictions, clip_processor, clip_model, image_path, device)

    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for pred in pred_boxes:
        x1, y1, x2, y2 = pred['bbox']
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{pred['class_id']} {pred['score']:.2f}", color='white', fontsize=12, weight='bold', backgroundcolor='red')
    
    # Save or display the result
    output_path = Path(output_dir) / f"{image.stem}_inference.jpg"
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Inference results saved to: {output_path}")

# Main function
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load models
    model, clip_processor, clip_model = initialize_models(device)
    
    # Run inference on input image(s)
    input_path = Path(args.input)
    
    if input_path.is_file():  # Single image
        run_inference(model, clip_processor, clip_model, input_path, args.output, device)
    elif input_path.is_dir():  # Directory of images
        for img_path in input_path.glob("*.jpg"):
            run_inference(model, clip_processor, clip_model, img_path, args.output, device)
    else:
        print(f"Invalid input path: {args.input}")

if __name__ == "__main__":
    main()
