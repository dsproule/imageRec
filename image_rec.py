import torch.nn as nn
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import time

class Model(nn.Module):
	def __init__(self, labels, id_=-1):
		super().__init__()

		# Initializes all the DNN stuff
		self.id_ = str(id_) if id_ != -1 else id_
		self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding=1)
		self.conv2 = nn.Conv2d(32, 216, kernel_size=2)
		self.conv3 = nn.Conv2d(216, 216, kernel_size=2)
		self.normalizer = nn.BatchNorm2d(216)
		self.dense1 = nn.Linear(5400, 216)
		self.dense2 = nn.Linear(216, 3)
		self.pool = nn.MaxPool2d(kernel_size=2)
		self.drop1 = nn.Dropout(.5)
		self.drop2 = nn.Dropout(.5)

		# Put all the layers in a list for easy looping
		self.conv_layers = [self.conv1, self.conv2, self.conv3]
		self.dense_layers = [self.dense1, self.dense2]
		self.drop_layers = [self.drop1, self.drop2]

		self.loss = nn.CrossEntropyLoss()
		self.labels = labels

		for file in os.listdir('models/'):
			dashes = [i for i, v in enumerate(file) if v == '-']
			if file[dashes[0]+1:dashes[1]] == self.id_:
				break
		else:
			# Block used to save results when id_ is set
			if self.id_ != -1:
				with open(f'models/autogen-{id_}-20-epoch(0)-0.pt', 'w') as f:
					f.write("Initialized file")

	# Creates one forward pass of the NN
	def forward(self, x):
		for conv_layer in self.conv_layers:
			x = F.relu(conv_layer(x))
			x = self.pool(x)
		
		x = self.normalizer(x)

		# Converts the CNN to a DNN then runs the forward pass
		x = torch.flatten(x, 1)
		for dense_layer, dropout in zip(self.dense_layers, self.drop_layers):
			x = dropout(x)
			x = F.relu(dense_layer(x))	

		# Returns the results
		return x

	# Uses loaded parameters to predict an image
	def predict(self, x, supress=False):
		with torch.no_grad():
			self.eval()

			# Converts it to img convention
			img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)[:-20] / 255.0
			img = cv2.resize(img, (50, 50))
			img = torch.Tensor(img).to(device)

			output = self.forward(img.view(-1, 1, 50, 50))
			output = F.softmax(output, dim=1)
			ind = torch.argmax(output, dim=1).item()
			
			if supress == False:
				print(f'Model predicts: {self.labels[ind]} with {round(output[0][ind].item() * 100., 4)} confidence')
			
			# Returns classification result as well as the confidence percentage
			return (self.labels[ind], round(output[0][ind].item() * 100., 4))

	def accuracy(self, X, y):
		batch_size = X.shape[0] // 20 if X.shape[0] >= 40 else 20

		avg = []
		# Predicts the results with current parameters and returns the results as a percentage of correct guesses
		for batch_sp in range(0, len(X), batch_size + 1):
			self.zero_grad()
			cur_batch_x = X[batch_sp: batch_sp + batch_size].view(-1, 1, 50, 50)
			y_true = y[batch_sp: batch_sp + batch_size]

			y_pred = self.forward(cur_batch_x)
			y_pred = F.softmax(y_pred, dim=1)
			avg.append(np.mean(np.array((torch.argmax(y_pred, dim=1) == torch.argmax(y_true, dim=1)).cpu())))
		return np.mean(avg)

	# Helper function that makes the model learn
	def fit_model(self, batch_size, epochs, X, y, X_val, y_val, optimizer=None, display_rate=20):
		init_time = time.perf_counter()

		# Bails the program if an optimizer was not set
		if optimizer == None:
			raise 'Need to specify an optimizer'
		
		strikes = 0
		for epoch in range(epochs):
			for batch_sp in range(0, len(X), batch_size):
				self.zero_grad()
				# Splits the data into batches
				cur_batch_x = X[batch_sp: batch_sp + batch_size].view(-1, 1, 50, 50)
				cur_batch_y = y[batch_sp: batch_sp + batch_size]

				output = self.forward(cur_batch_x)
				
				loss = self.loss(output, cur_batch_y)

				loss.backward()
				optimizer.step()
			if epoch % display_rate == 0:
				# Disables the dropout layer's deactivation
				self.eval()
				acc = 100 * self.accuracy(X_val, y_val)
				# If 60 percent of  the epochs have been completed and the accuracy is still sub 60% the model gains a strike
				# 	then if 5 strikes are accumulated the model leaves
				if epoch > .6 * epochs and acc < 60:
					strikes += 1
				if strikes >= 5:
					print("Model was bad, training aborted")
					return
				for file in os.listdir('models/'):
					# if the model didn't fail for being bad results
					dashes = [i for i, v in enumerate(file) if v == '-']
					if file[dashes[0]+1:dashes[1]] == self.id_:

						# Parses the naming convention of the model parameters
						last_acc = float(file[dashes[3]+1:-3])
						last_loss = float(file[dashes[1]+1:dashes[2]])

						# Replaces the old one if the current results are better
						if acc > last_acc and loss < last_loss:
							os.remove(f"models/{file}")
							self.save_model(f'models/autogen-{self.id_}-{round(loss.item(), 4)}-epoch({epoch})-{round(acc, 4)}.pt')
				# Reactivates the training
				self.train()
				print(f'epoch: {epoch}, loss: {loss}, time elapsed: {round(time.perf_counter() - init_time, 4)}s, \
validation accuracy: <{acc}>, id: {self.id_}')

	def save_model(self, name='model.pt'):
		torch.save(self.state_dict(), name)

	def load_model(self, name):
		# Name is a saved model's directory, path Ex. (ex/model.pt)
		try:
			self.load_state_dict(torch.load(name))
		except:
			self.load_state_dict(torch.load(name, map_location=torch.device('cpu')))

if __name__ == '__main__':
	# Tests the accuracy of the current model by loading in images0
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print("Running on %s..." % device)

	'''Loading the test sets into the thing'''
	full_set = np.load("ban_wat_app_train_set.npy", allow_pickle=True)
	val_set = np.load("ban_wat_app_val_set.npy", allow_pickle=True)

	X_train = torch.Tensor(np.array([i[0] for i in full_set])).to(device)
	y_train = torch.Tensor(np.array([i[1] for i in full_set])).to(device)

	X_val = torch.Tensor(np.array([i[0] for i in val_set])).to(device)
	y_val = torch.Tensor(np.array([i[1] for i in val_set])).to(device)

	'''To test how well it trained'''
	exts = ['png', 'jpg', 'jpeg', 'webp']
	test_ims = [im for im in os.listdir('test_ims/') if im[im.rfind('.') + 1:] in exts]

	file = 'autogen-0.001_0.5_0.01-0.023-epoch(440)-93.0759.pt'

	print("Beginning Tests...")
	correct_c = 0
	model = 'autogen-0.001_0.5_0.01-0.023-epoch(440)-93.0759.pt'
	net = Model({0: 'banana', 1: 'watermelon', 2: 'apple'}).to(device)
	net.load_model('models/'+model)
	for im in test_ims:
		res, conf = net.predict('test_ims/'+im, supress=True)
		if res[0] == im[0] and conf > 60:
			correct_c += 1
		else:
			print("Mistake at:", res, im, file, conf)
	print(f'{model}: {correct_c} / {len(test_ims)}')