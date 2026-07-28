from typing import List, Tuple

from pydantic.dataclasses import dataclass

from kumorfm.api.pquery.AST.ast_node import ASTNode
from kumorfm.api.pquery.AST.location_interval import ASTQueryLocationInterval


@dataclass(repr=False)
class Column(ASTNode):
    r"""Creates an atomic description of a column :obj:`fqn`.
    Args:
        fqn: A fully-qualified name in the "table.column" format.
    """

    fqn: str = ''

    def __post_init__(self) -> None:
        if len(self.fqn.split('.')) != 2:
            raise ValueError(
                f'The column name {self.fqn} was not given in its '
                f'fully-qualified name form. Format it as "table.column".')

        # internal helper var for checking if wildcard columns correctly
        # appear only inside COUNT aggregations
        self._count_col = False

        super().__post_init__()

    @property
    def all_query_columns_with_locations(
            self) -> List[Tuple[str, ASTQueryLocationInterval]]:
        assert self.location is not None
        return [(self.fqn, self.location)]

    @property
    def all_query_columns(self) -> List[str]:
        return [self.fqn]

    def to_string(self, rich: bool = False) -> str:
        return self.fqn
