import os
import yaml
import numpy as np
import copy

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

    def generate_options(filename: str, overrides: str):

        baseline = Settings.load_from_file(filename)

        with open(overrides) as overrides_file:
            overrides_data = yaml.full_load(overrides_file)

        options = []

        def _generate_options_helper(data, stack):
            if isinstance(data, list):
                options.append((tuple(stack), data))
                return
            
            print(data)
            for element in data:
                _generate_options_helper(data[element], stack + [element])
        
        _generate_options_helper(overrides_data, [])

        option_counts = [len(o[1]) for o in options]

        all_index_options = tuple(np.arange(o) for o in option_counts)

        all_idx_combos = np.array(np.meshgrid(*all_index_options)).T.reshape(-1,len(all_index_options))

        print("Options:",options)
        attr_stacks = [o[0] for o in options]

        settings_options = []
        settings_descriptions = []
        for idx_combo in all_idx_combos:
            settings_copy = copy.deepcopy(baseline)
            settings_description = ""
            for attr_idx, (option_idx, attr_stack) in enumerate(zip(idx_combo, attr_stacks)):
                element = settings_copy
                for attr in attr_stack[:-1]:
                    element = element[attr]
                attr_val = options[attr_idx][1][option_idx]
                element[attr_stack[-1]] = attr_val

                attr_path = ".".join(attr_stack)
                settings_description += f"{attr_path}={attr_val}\n"
            settings_options.append(settings_copy)
            settings_descriptions.append(settings_description)

        return settings_options, settings_descriptions