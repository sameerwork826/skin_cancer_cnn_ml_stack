import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, metadata_csv, images_dir, transform=None):
        self.meta = pd.read_csv(metadata_csv)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_id'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Prefer an encoded integer label column if present, otherwise fall back to string dx.
        if 'label_enc' in self.meta.columns:
            label = int(row['label_enc'])
        else:
            # return the raw diagnosis string (e.g., 'mel') if no encoding available
            # calling code should handle mapping to integers when needed
            label = row['dx']
        return image, label, row.to_dict()
