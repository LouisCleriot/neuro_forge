import torch
from src.layers.attention.selfAttention import SelfAttentionTextBook

def main():
    embedingSize = 1024
    attentionSize = 128
    selfAttention = SelfAttentionTextBook(embedingSize, attentionSize, "xavier")
    token_len = 5
    x = torch.randn(1, token_len, embedingSize)
    mask = torch.tril(torch.ones(token_len, token_len))
    out = selfAttention(x, mask)


if __name__ == "__main__":
    main()
