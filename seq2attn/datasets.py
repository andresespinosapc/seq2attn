import random
from copy import deepcopy
import torchtext

class TabularAugmentedDataset(torchtext.data.TabularDataset):
    def __getitem__(self, idx):
        current_example = super().__getitem__(idx)
        random_example = super().__getitem__(random.randint(0, self.__len__() - 1))
        final_example = deepcopy(current_example)
        final_example.src = current_example.src + random_example.src[:-1]

        return final_example
