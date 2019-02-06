import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_proportion=0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_proportion)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
        return F.log_softmax(x, dim=1)
    
    
def validation(model, test_loader, criterion):
    test_loss = 0
    accuracy = 0
    
    for images, labels in test_loader:
        images.resize_(images.shape[0], 784)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        predictions = torch.exp(output)
        equality = (labels.data == predictions.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy



def train(model, train_loader, test_loader, criterion, optimizer, epochs=5, print_every=50):
    steps = 0
    running_loss = 0
    
    for epoch in range(1, epochs + 1):
        model.train()  # Set to train mode
        for images, labels in train_loader:
            steps += 1

            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()  # Sets the network to eval mode

                with torch.no_grad():
                    test_loss, accuracy = validation(model, test_loader, criterion)

                print(f'Epoch: {epoch}/{epochs}...')
                print(f'Training Loss: {running_loss/print_every}')
                print(f'Test Loss: {test_loss/len(test_loader)}')
                print(f'Test Accuracy: {accuracy/len(test_loader)}')


                running_loss = 0
                model.train()