from pathlib import Path
from PIL import Image
from torch.utils.data.dataset import Dataset


class OCRDataset(Dataset):
    def __init__(self, image_transform=None):
        super(OCRDataset, self).__init__()
        self.paths = sorted(list(Path('./data_filtered/').glob('**/*.txt')))
        self.image_transform = image_transform

    def __getitem__(self, idx):
        txt_path = self.paths[idx]
        with open(txt_path, 'rt') as f:
            label = f.readline().strip()
            if label == '-':
                label = '-'*7
        image_path = txt_path.with_suffix('.png')
        image = Image.open(image_path)
        if self.image_transform:
            image = self.image_transform(image)
    
        return image, label, len(label)

    def __len__(self):
        return len(self.paths)

# if __name__ == '__main__':
#     dataset = OCRDataset()
#     for i, (image, label) in enumerate(dataset):
#         out_path = Path('data_filtered').joinpath(str(i))
#         image.save(str(out_path) + '.png')
#         with open(str(out_path) + '.txt', 'wt') as f:
#             f.write(label)
