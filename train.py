import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np


class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
 #           self.transform = transform
        def __getitem__(self, idx):
            label = self.labels[idx]
            image = self.images[idx]      
 #           image = self.transform(np.array(image))
            return image, label
        def __len__(self):
            return len(self.labels)
        

def train(model, n_epochs, train_images, train_labels):
    train_dataset = CustomDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, n_epochs+1):
      train_loss = 0.0
      model.train()
      for data, target in train_loader:
          # move tensors to GPU if CUDA is available
          if use_gpu:
               data, target = data.cuda(), target.cuda()
          # clear the gradients of all optimized variables
          optimizer.zero_grad()
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model(data)
          # calculate the batch loss
          loss = criterion(output, target)
          # backward pass: compute gradient of the loss with respect to model parameters
          loss.backward()
          # perform a single optimization step (parameter update)
          optimizer.step()
          # update training loss
          train_loss += loss.item()*data.size(0)
    
      print('Epoch: {} \tTraining Loss: {:.6f}'.format(
          epoch, train_loss))
    torch.save(model.state_dict(), 'model.pt')
    return model
