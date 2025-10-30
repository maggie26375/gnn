#!/usr/bin/env python3
"""
Merge trained decoder weights into the full model checkpoint.

This script is used after training the decoder separately to combine
the decoder weights with the main model checkpoint.

Usage:
    python merge_decoder_weights.py \
        --main_checkpoint checkpoints/last.ckpt \
        --decoder_checkpoint decoder_checkpoints/decoder-last.ckpt \
        --output checkpoints/merged_with_decoder.ckpt
"""
import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_decoder_weights(
    main_ckpt_path: str,
    decoder_ckpt_path: str,
    output_path: str,
):
    """
    Merge decoder weights from a decoder-only checkpoint into a full model checkpoint.

    Args:
        main_ckpt_path: Path to the main model checkpoint
        decoder_ckpt_path: Path to the decoder-only checkpoint
        output_path: Where to save the merged checkpoint
    """
    logger.info("="*60)
    logger.info("Merging Decoder Weights into Main Model")
    logger.info("="*60)

    # Load main checkpoint
    logger.info(f"\n1. Loading main model from: {main_ckpt_path}")
    main_ckpt = torch.load(main_ckpt_path, map_location='cpu', weights_only=False)
    logger.info(f"   Main checkpoint has {len(main_ckpt['state_dict'])} parameters")

    # Load decoder checkpoint
    logger.info(f"\n2. Loading decoder from: {decoder_ckpt_path}")
    decoder_ckpt = torch.load(decoder_ckpt_path, map_location='cpu', weights_only=False)
    logger.info(f"   Decoder checkpoint has {len(decoder_ckpt['state_dict'])} parameters")

    # Extract decoder weights from decoder checkpoint
    logger.info(f"\n3. Extracting decoder weights...")
    decoder_state_dict = {}
    for key, value in decoder_ckpt['state_dict'].items():
        if 'gene_decoder' in key:
            # Remove 'full_model.' prefix if present
            new_key = key.replace('full_model.', '')
            decoder_state_dict[new_key] = value
            logger.debug(f"   Found decoder param: {new_key}")

    logger.info(f"   ✅ Found {len(decoder_state_dict)} decoder parameters")

    # Update main checkpoint with new decoder weights
    logger.info(f"\n4. Updating main checkpoint with trained decoder weights...")
    updated_count = 0
    for key in decoder_state_dict:
        if key in main_ckpt['state_dict']:
            # Check if shape matches
            if main_ckpt['state_dict'][key].shape == decoder_state_dict[key].shape:
                main_ckpt['state_dict'][key] = decoder_state_dict[key]
                updated_count += 1
            else:
                logger.warning(f"   Shape mismatch for {key}: "
                             f"{main_ckpt['state_dict'][key].shape} vs {decoder_state_dict[key].shape}")
        else:
            logger.warning(f"   Parameter not found in main checkpoint: {key}")

    logger.info(f"   ✅ Updated {updated_count} parameters")

    # Save merged checkpoint
    logger.info(f"\n5. Saving merged checkpoint to: {output_path}")
    torch.save(main_ckpt, output_path)
    logger.info(f"   ✅ Merged checkpoint saved!")

    logger.info("\n" + "="*60)
    logger.info("✅ Merge Complete!")
    logger.info("="*60)
    logger.info(f"Main model: {main_ckpt_path}")
    logger.info(f"Decoder: {decoder_ckpt_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Updated {updated_count}/{len(decoder_state_dict)} decoder parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Merge trained decoder weights into full model checkpoint"
    )
    parser.add_argument(
        "--main_checkpoint",
        type=str,
        required=True,
        help="Path to main model checkpoint"
    )
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        required=True,
        help="Path to decoder-only checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged checkpoint"
    )

    args = parser.parse_args()

    merge_decoder_weights(
        main_ckpt_path=args.main_checkpoint,
        decoder_ckpt_path=args.decoder_checkpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
