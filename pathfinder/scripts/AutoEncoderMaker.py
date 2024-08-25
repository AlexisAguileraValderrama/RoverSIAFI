
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(output_dim/4)),  # Input layer -> Hidden layer
 
            nn.Linear(int(output_dim/4), int(output_dim/2)), # Hidden layer -> Latent space

            nn.Linear(int(output_dim/2), output_dim) # Latent space -> Bottleneck layer
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, int(output_dim/2)), # Bottleneck layer -> Latent space

            nn.Linear(int(output_dim/2), int(output_dim/4)),  # Latent space -> Hidden layer

            nn.Linear(int(output_dim/4), input_dim)    # Hidden layer -> Output layer
        )
    
    def forward(self, x):
        # Forward pass through the network
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fast_train(self):
        data_np = np.random.uniform(-2,2,(1000,self.input_dim))
        data = torch.from_numpy(data_np).float()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(),lr=0.001)

        num_epochs = 200
        for epoch in range(num_epochs):
            # Forward pass
            output = self(data)
            loss = criterion(output, data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training completed.")
    
    def encode(self, data_np):
        data = torch.from_numpy(data_np).float()
        return self.encoder(data).detach().numpy()
