import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class KonIQ10kDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.scores_frame = pd.read_csv(csv_path)
        self.img_dir = img_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
            
    def __len__(self):
        return len(self.scores_frame)
        
    def __getitem__(self, idx):
        img_name = self.scores_frame.iloc[idx]['image_name']
        mos_score = self.scores_frame.iloc[idx]['MOS']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Normalize MOS scores to [0,1] range
        mos_score = (mos_score - self.scores_frame['MOS'].min()) / \
                   (self.scores_frame['MOS'].max() - self.scores_frame['MOS'].min())
        
        return image, torch.tensor([mos_score], dtype=torch.float)
    

#FaceIQA dataset is defined here    
class FaceIQADataset(Dataset):
    def __init__(self, csv_path, root_dir):
        """
        Args:
            csv_path (str): Path to the CSV file with annotations.
            root_dir (str): Root directory for all face images (i.e., where Face_GT/ is located).
        """
        self.data_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        subject = row['SUBJECTNAMES'].strip()
        distance = row['DISTANCE'].strip()
        pose = row['POSE'].strip()
        filename = row['FILENAME'].strip()
        
        # Construct full image path
        img_path = os.path.join(self.root_dir, subject, distance, pose, filename)
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Assuming "Final score" is the target quality label
        final_score = row['FIQS']
        final_score = 0.0 if pd.isna(final_score) else final_score
        
        return image, torch.tensor([final_score], dtype=torch.float)

# Dataset configurations
DATASET_CONFIGS = {
    'koniq10k': {
        'csv_path': '/home/ug/Ganesh/Thesis/CBAM/koniq10k_1024x768/koniq10k_scores_and_distributions.csv',
        'image_dir': '/home/ug/Ganesh/Thesis/CBAM/koniq10k_1024x768/1024x768'
    },
    'kadid10k': {
        'csv_path': '/path/to/kadid10k/csv',
        'image_dir': '/path/to/kadid10k/images'
    },
    'faceiqa': {
    'csv_path': '/content/Dataset/Face_iqa/sorted_file.csv',
    'image_dir': '/content/Dataset/Face_iqa'
    }
}

def get_dataset(dataset_name):
    """
    Factory function to get dataset instance
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    if dataset_name == 'koniq10k':
        return KonIQ10kDataset(config['csv_path'], config['image_dir'])
    elif dataset_name == 'faceiqa':
        return FaceIQADataset(config['csv_path'], config['image_dir'])
    # Add other dataset classes as needed
    
    raise ValueError(f"Dataset {dataset_name} not implemented")
