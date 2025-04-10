import torch
from torchvision import transforms
from PIL import Image
import argparse
from vgg16_cbam import VGG16_CBAM_IQA
import os

class ImageQualityPredictor:
    def __init__(self, model_path, regression_type='simple'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = VGG16_CBAM_IQA(regression_type=regression_type).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict quality score for a single image
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        return prediction.item()
    
    def predict_batch(self, image_dir):
        """
        Predict quality scores for all images in a directory
        """
        results = {}
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                try:
                    score = self.predict(image_path)
                    results[filename] = score
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict image quality scores')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--regression_type', type=str, default='simple',
                      choices=['simple', 'elastic', 'ridge'],
                      help='Type of regression used in the model')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to image file or directory')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ImageQualityPredictor(
        model_path=args.model_path,
        regression_type=args.regression_type
    )
    
    # Make predictions
    if os.path.isfile(args.input):
        # Single image
        score = predictor.predict(args.input)
        print(f"Quality score for {args.input}: {score:.4f}")
    else:
        # Directory of images
        results = predictor.predict_batch(args.input)
        print("\nQuality Scores:")
        for filename, score in results.items():
            print(f"{filename}: {score:.4f}")

if __name__ == '__main__':
    main() 