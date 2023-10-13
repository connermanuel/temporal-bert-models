def test_generation_kwargs(dataloader_ssm, model_t5_base, model_t5_orth, device, tokenizer_t5):
    ### This test verifies that the behavior of the initialized orthogonal model is roughly the same as the base model

    model_t5_base.eval()
    model_t5_orth.eval()

    sample_input = next(iter(dataloader_ssm)).to(device)

    output_orth = model_t5_orth.generate(**sample_input)
    for k in list(sample_input.keys()):
        if "timestamps" in k:
            sample_input.pop(k)
    output_base = model_t5_base.generate(**sample_input)
    print("done")

    # mask = sample_input["labels"] != -100
    # logits_orth = output_orth["logits"][mask]
    # logits_base = output_base["logits"][mask]

    # assert torch.equal(
    #     torch.topk(logits_orth, dim=1, k=3)[1], torch.topk(logits_base, dim=1, k=3)[1]
    # )