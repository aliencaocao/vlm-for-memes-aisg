import torch

def merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, image_token_index=32000, pad_token_id=0):
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))
    # 1. Create a mask to know where special image tokens are
    special_image_token_mask = input_ids == image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    # Compute the maximum embed dimension
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged image-text sequence.
    # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
    # `torch.cumsum` computes how each image token shifts subsequent text token positions.
    # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)

    # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
    batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[batch_indices, pad_indices]

    final_embedding[batch_indices, indices_to_mask] = 0


    return final_embedding, final_attention_mask, position_ids
