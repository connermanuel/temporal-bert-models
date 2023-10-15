import torch

def test_generation(dataloader_ssm, model_t5_base, model_t5_orth, device):
    ### This test verifies that the behavior of the initialized orthogonal model is roughly the same as the base model

    model_t5_base.eval()
    model_t5_orth.eval()

    sample_input = next(iter(dataloader_ssm)).to(device)

    output_orth = model_t5_orth.generate(**sample_input)
    for k in list(sample_input.keys()):
        if "timestamps" in k:
            sample_input.pop(k)
    output_base = model_t5_base.generate(**sample_input)
    
    assert torch.equal(output_orth, output_base)
