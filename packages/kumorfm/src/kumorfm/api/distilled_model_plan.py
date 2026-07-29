# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field, fields
from typing import Dict, List, Union

from pydantic import ConfigDict
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

from kumorfm.api.model_plan import (
    ColumnProcessingPlan,
    Metadata,
    MissingType,
    OptimizationPlan,
    PlanMixin,
    RunMode,
    TrainingJobPlan,
    _add_indent,
    _SerializableColumnProcessingPlan,
    _SerializableOptimizationPlan,
    _SerializableTrainingJobPlan,
    compat_conlist,
    compat_field,
    confloat,
    conint,
)
from kumorfm.api.pquery import QueryType
from kumorfm.api.task import TaskType
from kumorfm.api.typing import WITH_PYDANTIC_V2, TimeUnit


@dataclass(config=dict(validate_assignment=True))
class TimeOffset:
    r"""Represents a time offset with a value and unit.

    :ivar value: (``int``) The numerical value of the time offset.
    :ivar unit: (``TimeUnit``) The unit of time for the offset
        (*default:* ``TimeUnit.DAYS``).
    """
    value: int = PydanticField(ge=0)
    unit: TimeUnit = compat_field(default=TimeUnit.DAYS, metadata=Metadata())


@dataclass(config=dict(validate_assignment=True))
class DistillationPlan:
    r"""Defines attributes that affect the features/interactions used to
    train the online serving model.

    :ivar embedding_keys: (``list[str]``) Key column(s) in the entity table
        used to extract embeddings from the deep model during distillation.
        A primary key extracts from the entity table itself; foreign key(s)
        extract from their corresponding 1-hop neighbor table(s).

    :ivar max_embedding_offset: (``TimeOffset``) Maximum staleness of deep
        model embeddings relative to the anchor time.
        Defines the upper bound on the offset between the embedding seed
        time and the anchor time.

    :ivar min_embedding_offset: (``TimeOffset``) Minimum staleness of deep
        model embeddings relative to the anchor time.
        Defines the lower bound on the offset between the embedding seed
        time and the anchor time. Models the latency between embedding
        generation and its availability at serving.

    :ivar real_time_interactions: (``dict[str, int]``) Real-time interaction
        key paths mapped to the maximum number of recent interactions to
        incorporate at inference time.
        For entity-level predictions, formatted as
        ``'entityTable.pkeyCol->interactionTable.fkeyCol'``.
        For fact-level predictions, formatted as
        ``'factTable.fkeyCol->entityTable.pkeyCol->interactionTable.fkeyCol'``.
        Examples: ``{'users.id->orders.user_id': 32}``,
        ``{'orders.user_id->users.id->views.user_id': 16}``.
        (*default:* ``{}``).

    :ivar real_time_offset: (``TimeOffset``) Minimum offset between the anchor
        time and the most recent real-time interaction available at serving.
        Models the end-to-end ingestion latency of the real-time data pipeline;
        interactions arriving within this window of the anchor time are
        excluded.

    :ivar trim_to_recent: (``bool``) Whether to trim the real-time interactions
        to the most recent interactions. If False, then real-time interactions
        before the max_embedding_offset are included.
        (*default:* ``True``).

    """
    embedding_keys: Union[List[str], MissingType] = compat_field(
        default_factory=lambda: MissingType.VALUE,
        metadata=Metadata(),
        min_length=1,
    )

    max_embedding_offset: Union[TimeOffset, MissingType] = compat_field(
        default=MissingType.VALUE,
        metadata=Metadata(),
    )

    min_embedding_offset: Union[TimeOffset, MissingType] = compat_field(
        default=MissingType.VALUE,
        metadata=Metadata(),
    )

    real_time_interactions: Dict[str, int] = compat_field(
        default_factory=dict,
        metadata=Metadata(),
    )

    real_time_offset: Union[TimeOffset, MissingType] = compat_field(
        default=MissingType.VALUE,
        metadata=Metadata(),
    )

    trim_to_recent: Union[bool, MissingType] = compat_field(
        default=MissingType.VALUE,
        metadata=Metadata(),
    )


@dataclass(config=dict(validate_assignment=True), repr=False)
class DistillationModelArchitecturePlan(PlanMixin):
    r"""Model architecture configuration for distilled models.

    :ivar channels: (``list[int]``) A list of candidate hidden
        feature dimensionalities for the online serving
        transformer model
        (*default:* ``[64, 128]``).

    :ivar num_layers: (``list[int]``) A list of candidate
        numbers of transformer layers
        (*default:* ``[4, 6]``).

    :ivar num_heads: (``list[int]``) A list of candidate
        numbers of attention heads
        (*default:* ``[8]``).

    :ivar emb_dropout: (``list[float]``) A list of candidate
        embedding dropout rates. Probability of zeroing out
        deep model embeddings during training
        (*default:* ``[0.0]``).

    :ivar dropout: (``list[float]``) A list of candidate
        dropout rates
        (*default:* ``[0.2]``).
    """
    channels: Union[
        compat_conlist(  # type: ignore
            conint(ge=1, le=512),
            min_length=1,
        ),
        MissingType] = compat_field(
            default_factory=lambda: MissingType.VALUE,
            metadata=Metadata(tunable=True),
        )

    num_layers: Union[
        compat_conlist(  # type: ignore
            conint(ge=1, le=12),
            min_length=1,
        ),
        MissingType] = compat_field(
            default_factory=lambda: MissingType.VALUE,
            metadata=Metadata(tunable=True),
        )

    num_heads: Union[
        compat_conlist(  # type: ignore
            conint(ge=1, le=32),
            min_length=1,
        ),
        MissingType] = compat_field(
            default_factory=lambda: MissingType.VALUE,
            metadata=Metadata(tunable=True),
        )

    emb_dropout: Union[
        compat_conlist(  # type: ignore
            confloat(ge=0.0, le=1.0),
            min_length=1,
        ),
        MissingType] = compat_field(
            default_factory=lambda: MissingType.VALUE,
            metadata=Metadata(tunable=True),
        )

    dropout: Union[
        compat_conlist(  # type: ignore
            confloat(ge=0.0, lt=1.0),
            min_length=1,
        ),
        MissingType] = compat_field(
            default_factory=lambda: MissingType.VALUE,
            metadata=Metadata(tunable=True),
        )


if WITH_PYDANTIC_V2:
    from pydantic import SerializeAsAny

    _SerializableDistillationPlan = SerializeAsAny[DistillationPlan]
    _SerializableDistillationModelArchitecturePlan = SerializeAsAny[
        DistillationModelArchitecturePlan]
else:
    _SerializableDistillationPlan = DistillationPlan
    _SerializableDistillationModelArchitecturePlan = (
        DistillationModelArchitecturePlan)


@dataclass(config=ConfigDict(validate_assignment=True), repr=False)
class DistilledModelPlan:
    r"""A complete definition of a Kumo distilled model plan.

    Encompasses a
    :class:`~kumorfm.api.model_plan.TrainingJobPlan`,
    :class:`~kumorfm.api.model_plan.ColumnProcessingPlan`,
    :class:`~kumorfm.api.model_plan.OptimizationPlan`,
    :class:`~kumorfm.api.distilled_model_plan.DistillationPlan`,
    and a :class:`~kumorfm.api.distilled_model_plan\
    .DistillationModelArchitecturePlan`.
    """
    training_job: _SerializableTrainingJobPlan = field(
        default_factory=TrainingJobPlan)
    # In the column_processing plan the training table
    # features can be keyed as `TRAIN_TABLE.{column_name}`.
    column_processing: _SerializableColumnProcessingPlan = field(
        default_factory=ColumnProcessingPlan)
    optimization: _SerializableOptimizationPlan = field(
        default_factory=OptimizationPlan)
    distillation: _SerializableDistillationPlan = field(
        default_factory=DistillationPlan)
    model_architecture: _SerializableDistillationModelArchitecturePlan = field(
        default_factory=DistillationModelArchitecturePlan)

    def __repr__(self) -> str:
        field_repr = '\n'.join(
            [f'{f.name}={getattr(self, f.name)},' for f in fields(self)])
        reprs = _add_indent(field_repr, num_spaces=2)
        return f'{self.__class__.__name__}(\n{reprs}\n)'


@dataclass
class SuggestDistilledModelPlanRequest:
    r"""A request to suggest a default distilled model plan based on
    ``query_string``, ``graph_id``, ``run_mode``, and ``base_model_id``.
    """
    query_string: str
    graph_id: str
    run_mode: RunMode
    base_model_id: str
    has_train_table_weight_col: bool = False


@dataclass
class DefaultDistilledModelPlanInfo:
    r"""A response containing the suggested distilled model plan and
    associated metadata.
    """
    distilled_model_plan: DistilledModelPlan
    task_type: TaskType
    query_type: QueryType
    has_train_table_weight_col: bool = False
