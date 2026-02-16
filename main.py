import torch
from src.layers.attention.selfAttention import SelfAttentionTextBook
from src.layers.attention.multiHeadAttention import MultiHeadAttention

def main():
    embedingSize = 1024
    attentionSize = 128
    selfAttention = MultiHeadAttention(embedingSize, 4, "xavier")
    token_len = 5
    x = torch.randn(1, token_len, embedingSize)
    x2 = torch.randn(1,1,embedingSize)
    mask = torch.tril(torch.ones(token_len, token_len))
    #mask2 = torch.tril(torch.ones())
    print(mask)
    out = selfAttention(x, mask)
    print("final shape :", out.size())
    out2 = selfAttention(x2)
    print("final 2", out2.size())

if __name__ == "__main__":
    main()
