import torch

TIMESTAMP_PAD = 2

def construct_time_matrix(self, original_layer: torch.Tensor, time_layers: nn.ModuleList, timestamps: torch.tensor):
    """
    Multiply rows of the input matrix with the corresponding time layers.

    Input:
        original_layer: tensor of shape (batch_size, n_heads, seq_length, dim_per_head)
        time_layers: Nested module list of shape (num_timestamps, num_attention_heads), each module is a linear layer
        timestamps: tensor of shape (batch_size, seq_len)
    Output:
        temporal_conditioned_layer: tensor of shape (batch_size, num_attention_heads, seq_len, attention_head_size)        
    """

    # TODO: Optimize
    timestamp_vals = torch.unique(timestamps[:, [0, -1]])
    masks = [torch.unsqueeze(timestamps==val, 1)[..., None] for val in timestamp_vals]
    temporal_conditioned_layers = [time_layers[val+TIMESTAMP_PAD](original_layer) * mask for val, mask in zip(timestamp_vals, masks)]
    temporal_conditioned_layer = torch.stack(temporal_conditioned_layers, dim=0).sum(dim=0)
    
    return temporal_conditioned_layer