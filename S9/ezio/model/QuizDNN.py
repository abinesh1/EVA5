import torch.nn as nn
import torch.nn.functional as F

class QuizDNN(nn.Module):
	def __init__(self):
		super(QuizDNN, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(3, 64, 3, padding = 1, bias = False),
			nn.ReLU(),
			nn.BatchNorm2d(64))
		self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding = 1, bias = False),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.Dropout(0.1))
		self.pool1 = nn.MaxPool2d(2,2)
		self.gap = nn.Sequential(nn.AvgPool2d(8),
			nn.BatchNorm2d(64))
		self.fully_conn = nn.Sequential(nn.Conv2d(64, 10, 1, padding = 0, bias = False))

	def forward(self, x):
		x1 = self.conv(x)
		x2 = self.conv2(x1)
		x3 = self.conv2(x1+x2)
		x4 = self.pool1(x1+x2+x3)
		x5 = self.conv2(x4)
		x6 = self.conv2(x4+x5)
		x7 = self.conv2(x4+x5+x6)
		x8 = self.pool1(x5+x6+x7)
		x9 = self.conv2(x8)
		x10 = self.conv2(x8+x9)
		x11 = self.conv2(x8+x9+10)
		x12 = self.gap(x11)
		x13 = self.fully_conn(x12)
		x = x13.view(-1, 10)

		return x
