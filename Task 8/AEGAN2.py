import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.autograd import Variable
# Load and preprocess the data
dataset = pd.read_csv(r'C:/Users/corey/Desktop/DOE sample data/trainingALLpreprocess3.csv')

# add noise
# noise_mean = 0
# noise_std = 0.015
# added_noise = np.random.normal(noise_mean, noise_std, size=data.shape)
# data = data + added_noise

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(dataset.T)

# Split the dataset into train and test sets
train_data = data_normalized#[:-1000]
AMI_resolution = 5
AMI_index = list(range(0, dataset.shape[0], AMI_resolution))


test_data = data_normalized[-1000:]
train_data_AMI = data_normalized.T[AMI_index].T
train_data_AMI_all = torch.tensor(train_data_AMI, dtype=torch.float32).cuda()
# Convert data to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32).cuda()
test_tensor = torch.tensor(test_data, dtype=torch.float32).cuda()

# Create DataLoader for train and test sets
train_loader = DataLoader(train_tensor, batch_size=1, shuffle=True)
train_loader_AMI = DataLoader(train_data_AMI_all, batch_size=1, shuffle=True)
test_loader = DataLoader(test_tensor, batch_size=1, shuffle=False)

# Autoencoder architecture

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 20),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded

# GAN architecture
class Generator(nn.Module):
    def __init__(self, latent_dim,output_dim):
        super(Generator, self).__init__()
        self.input_size = latent_dim
        self.hidden_size = output_dim

        self.fc1 = nn.Linear(latent_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



# class Generator(nn.Module):
#     def __init__(self, latent_dim,output_dim):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Model and optimizer initialization
input_dim = train_data.shape[1]
latent_dim = len(AMI_index)
output_dim = latent_dim

#autoencoder = Autoencoder(input_dim).cuda()

encoder = Encoder().cuda()
decoder = Decoder().cuda()
autoencoder = Autoencoder(encoder, decoder).cuda()

generator = Generator(latent_dim, output_dim).cuda()
discriminator = Discriminator(output_dim).cuda()

ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.00001)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)


g_optimizer = optim.Adam(generator.parameters(), lr=0.000001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)

scaler2 = MinMaxScaler()

ae_criterion = nn.L1Loss()
encoder_criterion = nn.L1Loss()
decoder_criterion = nn.L1Loss()
gan_criterion = nn.L1Loss()

ae_loss_list = []
encoder_loss_list = []

g_loss_list = []
d_loss_list = []
# Training loop

for epoch in range(10):
    for j, data in enumerate(zip(train_loader,train_loader_AMI)):
        encoder_optimizer.zero_grad()
        output = encoder(data[0])
        loss = encoder_criterion(output, data[1])
        encoder_loss_cpu = loss.cpu().detach().numpy()
        encoder_loss_list.append(encoder_loss_cpu)
        loss.backward()
        encoder_optimizer.step()


        if j % 100 == 0:
            print(f"Epoch [{epoch+1}/10] Batch [{j}/{len(train_loader)}] Encoder Loss: {loss.item():.4f}")

    #print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

for param in encoder.parameters():
    param.requires_grad = False

for epoch in range(10):
    for k, data in enumerate(zip(train_loader,train_loader_AMI)):
        ae_optimizer.zero_grad()
        encoded = encoder(data[0])
        output = autoencoder(data[0])

        loss = ae_criterion(output, data[0])
        ae_loss_cpu = loss.cpu().detach().numpy()
        ae_loss_list.append(ae_loss_cpu)
        loss.backward()
        ae_optimizer.step()
        if k % 100 == 0:
            print(f"Epoch [{epoch+1}/10] Batch [{k}/{len(train_loader)}] Autoencoder Loss: {loss.item():.4f}")

    #print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

epochs = 10
for epoch in range(epochs):
    for i, data in enumerate(zip(train_loader,train_loader_AMI)):
        # # Train autoencoder
        # autoencoder.zero_grad()
        # #recon_data = autoencoder(data)
        # encoded, decoded = autoencoder(data[0])
        # ae_loss = ae_criterion(decoded, data[0])
        # ae_loss.backward()

        latent_data = encoder.model(torch.tensor(data[0].cpu().data.numpy()).cuda().float())
        latent_data_normalized = scaler2.fit_transform(latent_data.cpu().data.numpy().T).T
        latent_data_normalized_all = torch.tensor(latent_data_normalized).cuda()


        #for l, data_AMI in enumerate(train_loader_AMI):
        target = data[1]
        #encoder_loss = encoder_criterion(encoded, target)
        #encoder_loss.backward()
        #ae_optimizer.step()



        # Train generator
        generator.zero_grad()
        #ami = torch.tensor(train_data_AMI, dtype=torch.float32).cuda()
        #ami = torch.randn(data.size(0), latent_dim).cuda()

        #target = ami
        #encoder_loss = encoder_criterion(encoded, target)

        gen_data = generator(target)
        #gen_data = generator(ami)
        d_fake = discriminator(gen_data)
        g_loss = gan_criterion(d_fake, torch.ones(latent_data_normalized_all.size(0), 1).cuda())
        g_loss_cpu = g_loss.cpu().detach().numpy()
        g_loss_list.append(g_loss_cpu)
        g_loss.backward()
        g_optimizer.step()


        # Train discriminator
        discriminator.zero_grad()
        d_real = discriminator(latent_data_normalized_all)
        d_fake = discriminator(gen_data.detach())  # Detach to avoid updating the generator
        d_loss_real = gan_criterion(d_real, torch.ones(latent_data_normalized_all.size(0), 1).cuda())
        d_loss_fake = gan_criterion(d_fake, torch.zeros(latent_data_normalized_all.size(0), 1).cuda())
        d_loss = (d_loss_real + d_loss_fake)/2
        d_loss_cpu = d_loss.cpu().detach().numpy()
        d_loss_list.append(d_loss_cpu)
        d_loss.backward()
        d_optimizer.step()

        for epoch in range(10):
            for j, data in enumerate(zip(train_loader, train_loader_AMI)):
                encoder_optimizer.zero_grad()
                output = encoder(data[0])
                loss = encoder_criterion(output, data[1])
                encoder_loss_cpu = loss.cpu().detach().numpy()
                encoder_loss_list.append(encoder_loss_cpu)
                loss.backward()
                encoder_optimizer.step()

                if j % 100 == 0:
                    print(f"Epoch [{epoch + 1}/10] Batch [{j}/{len(train_loader)}] Encoder Loss: {loss.item():.4f}")

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        # Log losses
    # if (epoch + 1) % 10 == 0:
    #     print(
    #         f'Epoch [{epoch + 1}/{epochs}] - Autoencoder Loss: {ae_loss.item():.4f} - Generator Loss: {g_loss.item():.4f} - Discriminator Loss: {d_loss.item():.4f}')

plt.figure(figsize=(6,4))
plt.plot(encoder_loss_list, label='Encoder Loss')
plt.plot(ae_loss_list, label='Autoencoder Loss')
plt.xlabel('Number of Iterations')
plt.ylabel('MSE')
plt.legend()
plt.show()



plt.figure(figsize=(6,4))
plt.plot(d_loss_list, label='Discriminator Loss')
plt.plot(g_loss_list, label='Generator Loss')
plt.xlabel('Number of Iterations')
plt.ylabel('MSE')
plt.legend()
plt.show()
#Evaluate and compare the original data and generated data


test_data_denorm = scaler.inverse_transform(np.array(test_data))

#Test autoencoder
test_reconstructions = []
for data1 in test_loader:
    with torch.no_grad():
        #recon_data = autoencoder(data1)
        decoded = autoencoder(data1)
        test_reconstructions.extend(decoded.cpu().numpy())

test_reconstructions = scaler.inverse_transform(np.array(test_reconstructions))

#Test generator
# test_generated = []
# for _ in range(len(test_data) // 64 + 1):
#     with torch.no_grad():
#         noise = torch.randn(64, latent_dim).cuda()
#         gen_data = generator(noise)
#         test_generated.extend(gen_data.cpu().numpy())
test_data = data_normalized[-1000:]
test_reconstructed = autoencoder(torch.tensor(test_data, dtype=torch.float32).cuda().float())
test_reconstructed_orginal = scaler.inverse_transform(np.array(test_reconstructed.cpu().detach()))
AMI_resolution2 = 5
AMI_index2 = list(range(0, dataset.shape[0], AMI_resolution2))
test_data_AMI = test_data.T[AMI_index2].T
test_data_denorm = scaler.inverse_transform(np.array(test_data))
generated_data = generator(torch.tensor(test_data_AMI,dtype=torch.float32).cuda())
generated_data_denorm1 = scaler2.inverse_transform(np.array(generated_data.cpu().detach().numpy()))
high_AMI_generated = autoencoder.decoder(torch.tensor(generated_data.cpu().data.numpy()).cuda().float())
generated_data_denorm = scaler.inverse_transform(np.array(high_AMI_generated[:len(high_AMI_generated)].cpu().detach().numpy()))
#test_generated = scaler.inverse_transform(np.array(test_generated[:len(test_data)]))
feature_idx = 0

plt.figure(figsize=(6,4))
plt.plot(test_data_denorm[:, feature_idx], label='Original Data')
plt.plot(test_reconstructed_orginal[:, feature_idx], label='Reconstructed Data (Autoencoder)')
plt.plot(generated_data_denorm[:, feature_idx], label='Generated Data (GAN)')
plt.xlabel('Number of samples')
plt.ylabel('Voltage Magnitude')
plt.legend()
plt.show()

#Plot original data, autoencoder reconstructions, and GAN-generated data
fig, ax = plt.subplots(3, 1, figsize=(15, 12))


#ax.set_yticks(y_ticks)
ax[0].plot(test_data_denorm[:, feature_idx], label="Original Data")

ax[0].set_ylim(111, 116)
ax[0].set_title("Original Voltage Magnitude Data")
ax[0].set_xlabel("Number of samples")
ax[0].set_ylabel("Voltage Magnitude")

ax[1].plot(test_reconstructed_orginal[:, feature_idx], label="Reconstructed Data (Autoencoder)")

ax[1].set_ylim(111, 116)
ax[1].set_title("Autoencoder Reconstructed Voltage Magnitude Data")
ax[1].set_xlabel("Number of samples")
ax[1].set_ylabel("Voltage Magnitude")

ax[2].plot(generated_data_denorm[:, feature_idx], label="Generated Data (GAN)")

ax[2].set_ylim(111, 116)
ax[2].set_title("GAN Generated Voltage Magnitude Data")
ax[2].set_xlabel("Number of samples")
ax[2].set_ylabel("Voltage Magnitude")

plt.tight_layout()
plt.show()

a=1

# #Evaluation
# autoencoder.eval()
# reconstructed_data = []
#
# with torch.no_grad():
#     for data in test_loader:
#         recon_data = autoencoder(data)
#         reconstructed_data.append(recon_data.cpu().numpy())
#
# reconstructed_data = np.vstack(reconstructed_data)
# reconstructed_data = scaler.inverse_transform(reconstructed_data)
# original_test_data = scaler.inverse_transform(test_data)
#
#
# #Plotting original data and reconstructed data
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 6))
#
# #Plot original data
# plt.subplot(1, 2, 1)
# plt.plot(original_test_data)
# plt.title("Original Test Data")
# plt.xlabel("Sample Index")
# plt.ylabel("Voltage Magnitude")
#
# #Plot reconstructed data
# plt.subplot(1, 2, 2)
# plt.plot(reconstructed_data)
# plt.title("Reconstructed Test Data")
# plt.xlabel("Sample Index")
# plt.ylabel("Voltage Magnitude")
#
# plt.tight_layout()
# plt.show()