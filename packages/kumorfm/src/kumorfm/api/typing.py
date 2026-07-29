# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins

import pydantic
from pandas import DateOffset

from kumorfm.api.common import StrEnum

WITH_PYDANTIC_V2 = int(pydantic.__version__.split('.')[0]) >= 2

if WITH_PYDANTIC_V2:
    from pydantic import field_validator as _compatible_field_validator
else:
    from pydantic import validator as _compatible_field_validator  # ty: ignore

compatible_field_validator = _compatible_field_validator


class Stype(StrEnum):
    r"""The semantic type of a column.

    A semantic type denotes the semantic meaning of a column, and determines
    the preprocessing that is applied to the column. Semantic types can be
    passed to methods in the SDK as strings (*e.g.* ``"numerical"``).

    .. note::

        For more information about how to select a semantic type, please
        refer to the column-preprocessing documentation.

    Attributes:
        numerical: A numerical column. Typically integers or floats.
        categorical: A categorical column. Typically boolean or string values
            typically a single token in length.
        multicategorical: A multi-categorical column. Typically a concatenation
            of multiple categories under a single string representation.
        ID: A column holding IDs. Typically numerical values used to uniquely
            identify different entities.
        text: A text column. String values typically multiple tokens in length,
            where the actual language content of the value has semantic
            meaning.
        timestamp: A date/time column.
        sequence: A column holding sequences/embeddings. Consists of lists of
            floats, all of equal length, and are typically the output of
            another AI model
        image: A column holding image URLs.
    """
    numerical = 'numerical'
    categorical = 'categorical'
    multicategorical = 'multicategorical'
    ID = 'ID'
    text = 'text'
    timestamp = 'timestamp'
    sequence = 'sequence'
    image = 'image'
    unsupported = 'unsupported'

    def to_parent_stype(self) -> 'Stype':
        r"""Convert the semantic type to its parent type.

        Most semantic types are their own parent type. However, ``ID`` is
        converted to ``categorical`` because it is a special case.
        """
        return self if self != Stype.ID else Stype.categorical

    def supports_dtype(self, dtype: 'Dtype') -> bool:
        r"""Whether a :class:`Stype` supports a :class:`Dtype`."""
        if self == Stype.numerical:
            return dtype.is_numerical()
        if self == Stype.categorical:
            return dtype.is_bool() or dtype.is_numerical() or dtype.is_string()
        if self == Stype.multicategorical:
            return dtype.is_string() or dtype.is_list()
        if self == Stype.ID:
            return dtype.is_int() or dtype.is_string() or dtype.is_float()
        if self == Stype.text:
            return dtype in {Dtype.string}
        if self == Stype.timestamp:
            return dtype.is_maybe_timestamp()
        if self == Stype.sequence:
            return dtype in {
                Dtype.floatlist,
                Dtype.intlist,
                Dtype.string,
            }
        if self == Stype.image:
            return dtype in {Dtype.string}

        assert self == Stype.unsupported
        return True


class Dtype(StrEnum):
    r"""The data type of a column.

    A data type represents how the data of a column is physically stored. Data
    types can be passed to methods in the SDK as strings (*e.g.* ``"int"``).

    Attributes:
        bool: A boolean column.
        int: An integer column.
        float: An floating-point column.
        date: A column holding a date.
        time: A column holding a timestamp.
        floatlist: A column holding a list of floating-point values.
        intlist: A column holding a list of integers.
        binary: A column containing binary data.
        stringlist: A column containing list of strings.
    """
    # Booleans:
    bool = 'bool'
    # Integers:
    int = 'int'
    byte = 'byte'
    int16 = 'int16'
    int32 = 'int32'
    int64 = 'int64'
    # Floating point numbers:
    float = 'float'
    float32 = 'float32'
    float64 = 'float64'
    # Strings:
    string = 'string'
    binary = 'binary'
    # Time:
    date = 'date'
    time = 'time'
    timedelta = 'timedelta'
    # Nested lists:
    floatlist = 'floatlist'
    intlist = 'intlist'
    stringlist = 'stringlist'
    # Unsupported:
    unsupported = 'unsupported'

    def is_bool(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds booleans."""
        return self in {Dtype.bool}

    def is_int(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds integers."""
        return self in {
            Dtype.int, Dtype.byte, Dtype.int16, Dtype.int32, Dtype.int64
        }

    def is_float(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds floating point numbers."""
        return self in {Dtype.float, Dtype.float32, Dtype.float64}

    def is_numerical(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds numbers."""
        return self.is_int() or self.is_float() or self == Dtype.timedelta

    def is_string(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds strings."""
        return self in {Dtype.string, Dtype.binary}

    def is_timestamp(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds timestamps."""
        return self in {Dtype.date, Dtype.time}

    def is_maybe_timestamp(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds castable timestamps."""
        return self.is_timestamp() or self in {Dtype.string}

    def is_list(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds nested lists."""
        return self in {Dtype.floatlist, Dtype.intlist, Dtype.stringlist}

    def is_unsupported(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds unsupported types."""
        return self in {Dtype.unsupported}

    @property
    def default_stype(self) -> Stype:
        r"""Returns the default semantic type of this data type."""
        if self.is_bool():
            return Stype.categorical
        if self.is_numerical():
            return Stype.numerical
        if self == Dtype.binary:
            return Stype.categorical
        if self == Dtype.string:
            return Stype.text
        if self.is_timestamp():
            return Stype.timestamp
        if self in {Dtype.stringlist}:
            return Stype.multicategorical
        if self in {Dtype.floatlist, Dtype.intlist}:
            return Stype.sequence

        assert self == Dtype.unsupported
        return Stype.unsupported


class ColStatType(StrEnum):
    # Any:
    COUNT = 'COUNT'
    NUM_NA = 'NUM_NA'
    NA_FRACTION = 'NA_FRACTION'
    INVALID_FRACTION = 'INVALID_FRACTION'

    # Numerical, Temporal
    MIN = 'MIN'
    MAX = 'MAX'

    # Numerical:
    MEAN = 'MEAN'
    QUANTILES = 'QUANTILES'
    QUANTILE25 = 'QUANTILE25'
    MEDIAN = 'MEDIAN'
    QUANTILE75 = 'QUANTILE75'
    STD = 'STD'
    KURTOSIS = 'KURTOSIS'
    HISTOGRAM = 'HISTOGRAM'
    # num irrational entries (which are included in NA count and treated as NA)
    NUM_IRRATIONAL = 'NUM_IRRATIONAL'

    # Categorical:
    # NUM_UNIQUE and NUM_UNIQUE_MULTI count empty strings / NA values as their
    # own category. CATEGORY_COUNTS and MULTI_CATEGORY_COUNTS do not include
    # empty strings / NA values as their own category.
    NUM_UNIQUE = 'NUM_UNIQUE'
    NUM_UNIQUE_MULTI = 'NUM_UNIQUE_MULTI'
    CATEGORY_COUNTS = 'CATEGORY_COUNTS'
    MULTI_CATEGORY_COUNTS = 'MULTI_CATEGORY_COUNTS'

    UNIQUE_FRACTION = 'UNIQUE_FRACTION'

    # The separator to use for the multi-categorical column:
    MULTI_CATEGORIES_SEPARATOR = 'MULTI_CATEGORIES_SEPARATOR'

    # Strings:
    STRING_AVG_LEN = 'STRING_AVG_LEN'
    STRING_MAX_LEN = 'STRING_MAX_LEN'
    STRING_AVG_TOKENS = 'STRING_AVG_TOKENS'
    STRING_MAX_TOKENS = 'STRING_MAX_TOKENS'
    STRING_GLOVE_OVERLAP = 'STRING_GLOVE_OVERLAP'
    STRING_AVG_NON_CHAR = 'STRING_AVG_NON_CHAR'
    STRING_ARR_MIN_LEN = 'STRING_ARR_MIN_LEN'
    STRING_ARR_MAX_LEN = 'STRING_ARR_MAX_LEN'

    # Sequence:
    SEQUENCE_MAX_LENGTH = 'SEQUENCE_MAX_LENGTH'
    SEQUENCE_MIN_LENGTH = 'SEQUENCE_MIN_LENGTH'
    SEQUENCE_MEAN = 'SEQUENCE_MEAN'
    SEQUENCE_STD = 'SEQUENCE_STD'


class TimeUnit(StrEnum):
    r"""Defines the unit of a time."""
    SECONDS = 'seconds'
    MINUTES = 'minutes'
    HOURS = 'hours'
    DAYS = 'days'
    WEEKS = 'weeks'
    MONTHS = 'months'

    def to_offset(self) -> DateOffset:
        return DateOffset(**{self: 1})


class ProblemType(StrEnum):
    r"""Defines supported problem types.
    RANK, CLASSIFY, and FORECAST are supported.
    With RANK we internally use a ranking loss while training
    and during batch prediction we output top k targets.
    With CLASSIFY we use a classification loss.
    With FORECAST we use a time-series forecasting approach.
    """
    RANK = 'RANK'
    CLASSIFY = 'CLASSIFY'
    FORECAST = 'FORECAST'


class AggregationType(StrEnum):
    r"""Defines supported aggregations."""
    SUM = 'SUM'
    AVG = 'AVG'
    MIN = 'MIN'
    MAX = 'MAX'
    COUNT = 'COUNT'
    COUNT_DISTINCT = 'COUNT_DISTINCT'
    FIRST = 'FIRST'
    LAST = 'LAST'
    LIST_DISTINCT = 'LIST_DISTINCT'


class RelOp(StrEnum):
    r"""Defines relational operators: :obj:`!=, <=, >=, =, <, >`."""
    NEQ = '!='
    LEQ = '<='
    GEQ = '>='
    EQ = '='
    LT = '<'
    GT = '>'


class MemberOp(StrEnum):
    r"""Defines membership operators: :obj:`IS_IN`."""
    IS_IN = 'IS IN'
    IN = 'IN'


class StrOp(StrEnum):
    r"""Defines string operators: :obj:`STARTS_WITH, ENDS_WITH, CONTAINS,
    NOT_CONTAINS`."""
    STARTS_WITH = 'STARTS WITH'
    ENDS_WITH = 'ENDS WITH'
    CONTAINS = 'CONTAINS'
    NOT_CONTAINS = 'NOT CONTAINS'


class BoolOp(StrEnum):
    r"""Defines boolean operators: :obj:`AND, OR, NOT`."""
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
