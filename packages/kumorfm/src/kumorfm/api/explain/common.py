from dataclasses import field
from typing import Dict, List

from pydantic.dataclasses import dataclass


@dataclass
class EntityMappings:
    r"""Stores the mappings between each entity to the category
    it belongs to, such as 'true_positive', 'random' etc.

    Args:
        mappings (Dict[str, List[str]]): The mappings of
            selection strategy/category and corresponding entity ids
            starting from zero. These ids are used as file names for
            the entity-level XAI files.
        original_mappings (Dict[str, List[str]]): The mappings of
            selection strategy/category and corresponding original
            entity primary keys.
    """
    mappings: Dict[str, List[str]] = field(default_factory=dict)
    orig_mappings: Dict[str, List[str]] = field(default_factory=dict)
