# utils/dataloader_utils.py

from typing import Union, List, Tuple


def parse_batch_size(val: Union[int, List[int], Tuple[int, ...]]) -> int:
    """
    Safely parse batch_size whether defined in YAML configuration
    as an integer (e.g. 4) or a list/tuple (e.g. [4]).
    """
    if isinstance(val, (list, tuple)):
        return int(val[0])
    return int(val)


# Backward compatible alias
_parse_batch_size = parse_batch_size
