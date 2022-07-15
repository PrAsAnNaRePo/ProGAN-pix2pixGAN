from matplotlib import pyplot as plt
import torch
from torch import batch_norm, nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

# To increase some performance.
torch.backends.cudnn.benchmark = True

# path to the data
DATASET_DIR = "D:/dataset/resolution"

# HYPERPARAMETERS
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 30

# Preparing data augumentation.
transforms = A.Compose([
	A.RandomBrightnessContrast(p=0.8),
	A.Resize(16, 16),
	A.RandomFog(p=0.7)
])

# let's create the dataset class!
class Data(Dataset):
	def __init__(self, path, transforms=None):
		super().__init__()
		self.dir = path
		self.transforms = transforms
		self.images = os.listdir(self.dir)
	
	def __len__(self):
		return 5000# limiting the no. of images, if you have enough vram you can use 'len(self.images)'

	def __getitem__(self, index):
		image = np.array(Image.open(f'{self.dir}/{self.images[index]}').convert('RGB'), dtype='float32')
		x = self.transforms(image=image)['image']
		return x.reshape(3, 16, 16), image.reshape(3, 512, 512)

data = Data(DATASET_DIR, transforms)
# print(len(data), data[0][0].shape, data[0][1].shape)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# creating models...

class GeneratorBlock(nn.Module):
	def __init__(self, size, in_ch, out_ch):
		super().__init__()
		self.block = nn.Sequential(
				nn.UpsamplingNearest2d((size, size)),
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
				nn.LeakyReLU(),
				nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
				nn.LeakyReLU(),
			)
	
	def forward(self, x):
		return self.block(x)

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.block1 = GeneratorBlock(32, 3, 512) # N, 512, 32, 32
		self.block2 = GeneratorBlock(64, 512, 256) # N, 256, 64, 64
		self.block3 = GeneratorBlock(128, 256, 128) # N, 128, 128, 128
		self.block4 = GeneratorBlock(256, 128, 64) # N, 64, 256, 256
		self.block5 = GeneratorBlock(512, 64, 32)
		self.block6 = nn.Sequential(
				nn.Dropout(0.48),
				nn.Conv2d(32, 3, kernel_size=3, padding=1),
			)
		self.dropout = nn.Dropout(0.34)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.dropout(self.block3(x))
		x = self.block4(x)
		x = self.block5(x)
		return self.block6(x)

gen = Generator().to('cuda')
# print(gen(torch.rand((1, 3, 16, 16))).shape)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 2, 2),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 1, 1),
			nn.LeakyReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, 2, 2),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
			nn.Conv2d(128, 128, 1, 1),
			nn.LeakyReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(128, 256, 2, 2),
			nn.LeakyReLU(),
			nn.Conv2d(256, 256, 1, 1),
			nn.LeakyReLU(),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(256, 512, 2, 2),
			nn.LeakyReLU(),
			nn.Dropout(0.48),
			nn.Conv2d(512, 512, 1, 1),
			nn.LeakyReLU(),
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(512, 32, 2, 2),
			nn.LeakyReLU(),
			nn.Conv2d(32, 3, 1, 1),
			nn.LeakyReLU(),
		)

		self.conv6 = nn.Sequential(
			nn.Conv2d(6, 1, 1)
		)

	def forward(self, x, y):
		y = self.conv1(y)
		y = self.conv2(y)
		y = self.conv3(y)
		y = self.conv4(y)
		y = self.conv5(y)
		cat = torch.cat([x, y], 1)
		x = self.conv6(cat)
		return x

disc = Discriminator().to('cuda')
# print(disc(torch.rand((1, 3, 16, 16)), torch.rand((1, 3, 512, 512))).shape)
# exit()
gen_optim = torch.optim.Adam(gen.parameters(), lr=LR)
disc_optim = torch.optim.Adam(disc.parameters(), lr=LR)

# Here we are using two loss functions
BCELoss = nn.BCEWithLogitsLoss()
L1Loss = nn.L1Loss()

# TO train model in float16
g_scaler = torch.cuda.amp.grad_scaler.GradScaler()
d_scaler = torch.cuda.amp.grad_scaler.GradScaler()

# creating the training loop.
def train():
	for e in range(1, EPOCHS+1):
		print(f'[{e}/{EPOCHS}]=============>')
		loop = tqdm(loader)
		gen.train()
		disc.train()
		for i, (x, y) in enumerate(loop):
			x = x.to('cuda')
			y = y.to('cuda')
			
			with torch.cuda.amp.autocast():
				y_fake = gen(x)
				D_real = disc(x, y)
				D_real_loss = BCELoss(D_real, torch.ones_like(D_real))
				D_fake = disc(x, y_fake.detach())
				D_fake_loss = BCELoss(D_fake, torch.zeros_like(D_fake))
				D_loss = (D_real_loss + D_fake_loss) / 2
				
			disc.zero_grad()
			d_scaler.scale(D_loss).backward()
			d_scaler.step(disc_optim)
			d_scaler.update()

			with torch.cuda.amp.autocast():
				D_fake = disc(x, y_fake)
				G_fake_loss = BCELoss(D_fake, torch.ones_like(D_fake))
				L1 = L1Loss(y_fake, y) * 100
				G_loss = G_fake_loss + L1

			gen_optim.zero_grad()
			g_scaler.scale(G_loss).backward()
			g_scaler.step(gen_optim)
			g_scaler.update()
		
		gen.eval()
		data = x[0].reshape(1, 3, 16, 16)
		pred = gen(data)
		plt.imshow(pred.detach().cpu().numpy().reshape(512, 512, 3))
		plt.show()

if __name__ == '__main__':
	train()
	ck = {
		'model':gen,
		'state':gen.state_dict()
	}
