"""U-Net: predicts burn probability for every pixel of a tile at once.

Planned (phase 4): an encoder that downsamples the tile to capture landscape
context, and a decoder that upsamples back to full resolution, with skip
connections carrying fine detail across. Output is one logit per pixel, which
suits mapping burn probability across the whole grid.
"""
