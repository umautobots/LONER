import os
import yaml

from attrdict import AttrDict

# https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another


class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)

    def include(self, node):
        fname = os.path.join(self._root, self.construct_scalar(node))

        with open(fname, 'r') as f:
            return yaml.load(f, SettingsLoader)


class Settings(AttrDict):
    """ Settings class is a thin wrapper around AttrDict.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_from_file(filename: str):
        SettingsLoader.add_constructor('!include', SettingsLoader.include)

        with open(filename, 'r') as f:
            return Settings(yaml.load(f, SettingsLoader))
