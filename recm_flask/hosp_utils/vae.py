import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Variational Autoencoder 클래스
        :param input_dim: 입력 데이터 차원 (예: 784 for MNIST)
        :param hidden_dim: 중간층 (hidden layer) 차원
        :param latent_dim: 잠재 공간(latent space) 차원
        """
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)  # Mean vector
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)  # Log variance vector
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def encode(self, x):
        """
        Encoder를 통해 입력 데이터를 잠재 공간의 평균과 로그 분산으로 변환
        :param x: 입력 데이터
        :return: 잠재 공간의 평균(mu)과 로그 분산(log_var)
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        :param mu: 잠재 공간의 평균
        :param log_var: 잠재 공간의 로그 분산
        :return: 잠재 공간에서 샘플링된 벡터
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decoder를 통해 잠재 공간에서 입력 데이터로 복원
        :param z: 잠재 공간에서 샘플링된 벡터
        :return: 복원된 데이터
        """
        return self.decoder(z)

    def forward(self, x):
        """
        VAE 모델의 순전파
        :param x: 입력 데이터
        :return: 복원된 데이터, 평균(mu), 로그 분산(log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def loss_function(self, reconstructed, x, mu, log_var):
        """
        VAE 손실 함수: Reconstruction Loss + KL Divergence
        :param reconstructed: 복원된 데이터
        :param x: 원본 데이터
        :param mu: 잠재 공간의 평균
        :param log_var: 잠재 공간의 로그 분산
        :return: 총 손실 값
        """
        # Reconstruction Loss (Binary Cross-Entropy)
        recon_loss = F.binary_cross_entropy(reconstructed, x, reduction="sum")
        
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_div
