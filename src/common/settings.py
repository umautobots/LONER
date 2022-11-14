from attrdict import AttrDict


class Settings(AttrDict):
    """ Settings class is a simple wrapper around AttrDict.

    Left in its own class for future-proofing.
    For now, just use exactly as you'd use an AttrDict.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
