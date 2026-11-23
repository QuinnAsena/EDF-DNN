"""Network architectures. Each takes a batch of inputs and returns raw logits.

The experiment configs name an architecture by its class name
(`architecture: FeedforwardTwo`); get_architecture turns that name into the class.
"""

from edf_dnn.models.mlp import FeedforwardSingle, FeedforwardTapered, FeedforwardTwo

# Name -> class, like a named list in R: list(FeedforwardTwo = FeedforwardTwo, ...).
# New architectures (the CNN, the U-Net) get added to this list.
ARCHITECTURES = {model.__name__: model
                 for model in [FeedforwardSingle, FeedforwardTwo, FeedforwardTapered]}


def get_architecture(name):
    """The model class called `name`, e.g. "FeedforwardTwo" -> FeedforwardTwo."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name]
