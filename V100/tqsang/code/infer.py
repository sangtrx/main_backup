from model import OCR
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import torch, os
from torchvision import transforms
from random import randrange


'''
parser = ArgumentParser()
parser.add_argument('input_image')
args = parser.parse_args()
'''


device = 'cuda'
checkpoint = torch.load('./latest.pth', map_location=device)
vocab = checkpoint['vocab']

model = OCR(len(vocab))
model.eval()
model = model.to(device)
model.load_state_dict(checkpoint['model'])

stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}

image_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor()
])


num_samples = 0
correct_samples = 0
testPath = '/home/ngan_uark/tqsang/code/Train/Sequences'
image_names = []
predicts = []
for image_path in sorted(Path(testPath).glob('*.png')):
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0)
    image = image.to(device)
    out = model(image)
    out = out.squeeze(0).cpu()  # T, V
    out = out.argmax(-1)  # T
    '''
    with open(image_path.with_suffix('.txt'), 'rt') as f:
        label = f.readline().strip()
    '''
    pred = ''.join([itos[x.item()] for x in out])
    pred = pred[0] + ''.join([c for i, c in enumerate(pred[1:], 1) if c != pred[i-1]])
    pred = pred.replace('~', '')

    if pred == '':
        pred = randrange(100,999)

    image_names.append(os.path.basename(image_path))
    predicts.append(pred)
    '''
    if pred != label:
        print(pred, label)
    num_samples += 1
    correct_samples += 1 if pred == label else 0
    '''

assert len(image_names) == len(predicts)
for i in range(len(predicts)):
    with open('results.txt', 'a') as f:
        f.write(f"{image_names[i]}\t{predicts[i]}\n")

#print(f'ACCURACY {(correct_samples / num_samples) * 100:.02f}')
