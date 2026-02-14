from src.interfaces import SequenceMixer

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        mixer: SequenceMixer,
        mlp: nn.Module, 
        norm_layer: nn.Module
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = mixer
        self.norm2 = norm_layer(dim)
        self.mlp = mlp

    def forward(self, x, mask=None):
        x = x + self.mixer(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x
