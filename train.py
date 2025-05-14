import torch
import argparse
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from torch.utils.data import DataLoader
from dataset import TrashInWaterChannelsDataset  # Custom Dataset Class for loading the images and labels

# Argument parsing for customization
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with CLIP rescoring on Trash in Water Channels dataset.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data YAML file")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--img-size', type=int, default=416, help="Image size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on (cpu/cuda)")
    return parser.parse_args()

# Setup dataset
def setup_dataset(data_yaml, batch_size, img_size):
    # Create custom dataset and dataloaders
    train_dataset = TrashInWaterChannelsDataset(data_yaml, split="train", img_size=img_size)
    val_dataset = TrashInWaterChannelsDataset(data_yaml, split="val", img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Initialize model and CLIP processor
def initialize_models(device):
    model = YOLO("yolov8s.pt").to(device)  # Use the pre-trained YOLOv8s model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return model, clip_processor, clip_model

# Custom training loop
def train(model, clip_processor, clip_model, train_loader, val_loader, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass through YOLO
            optimizer.zero_grad()
            pred = model(imgs)
            loss = pred.loss  # Default loss in YOLO model

            # Apply CLIP rescoring on predicted boxes
            for idx, img in enumerate(imgs):
                img_emb = clip_model.get_image_features(**clip_processor(images=img, return_tensors="pt").to(device))
                # Rescore bounding boxes with CLIP features - Example method
                pred_boxes = pred.boxes[idx]  # Get predictions for the current image
                # Perform rescoring using CLIP features (fine-tune as needed)
                # Update model based on rescored predictions

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader)}")
        
        validate(model, val_loader, clip_processor, clip_model, device)  # Run validation

# Validation loop
def validate(model, val_loader, clip_processor, clip_model, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass through YOLO
            pred = model(imgs)
            loss = pred.loss
            val_loss += loss.item()

            # CLIP-based rescoring (same as training)
            for idx, img in enumerate(imgs):
                img_emb = clip_model.get_image_features(**clip_processor(images=img, return_tensors="pt").to(device))
                # Rescore bounding boxes with CLIP features

    print(f"Validation Loss: {val_loss / len(val_loader)}")

# Main function to run the training
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader = setup_dataset(args.data, args.batch_size, args.img_size)
    model, clip_processor, clip_model = initialize_models(device)

    # Optimizer setup for YOLOv8
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train(model, clip_processor, clip_model, train_loader, val_loader, optimizer, device, args.epochs)

if __name__ == "__main__":
    main()

