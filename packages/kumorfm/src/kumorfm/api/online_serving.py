from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import Field
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass

from kumorfm.api.common import StrEnum
from kumorfm.api.data_snapshot import GraphSnapshotID
from kumorfm.api.graph import Edge, TableName
from kumorfm.api.task import TaskType
from kumorfm.api.typing import WITH_PYDANTIC_V2


@dataclass
class OnlinePredictionOptions:
    # Required if prediction task is to perform binary classification.
    binary_classification_threshold: Optional[float] = 0.5

    # On classification tasks, for each entity, we will only return predictions
    # for the K classes with the highest predicted values for the entity.
    # If empty, predict all class. This field is ignored for regression tasks.
    num_classes_to_return: Optional[int] = None


# Request body for either launch or patch(update) an online serving endpoint:
#
# To launch one:
#    POST /online_serving_endpoints {OnlineServingEndpointRequest body}
#      => return {OnlineServingEndpointResource} + 200 (upon success)
# To update one:
#    PATCH /online_serving_endpoints/{id} {OnlineServingEndpointRequest body}
#      => return {OnlineServingEndpointResource} + 200 (upon success)
@dataclass
class OnlineServingEndpointRequest:
    """POST request body to create an Online Serving endpoint."""
    # ID of a (successful) model training job.
    model_training_job_id: str

    predict_options: OnlinePredictionOptions

    # Optional, a specific Graph data snapshot to be loaded for online
    # prediction. If this field is absent in the launch or update request,
    # this instructs Kumo to refresh the graph data and load the most
    # recently refreshed graph data snapshot for online serving.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Estimated max # of requests per second. This field can be useful for Kumo
    # to provision sufficient serving capacity and to configure rate limiting
    # and/or load shedding.
    max_qps: int = 50


class OnlineServingStatusCode(StrEnum):
    # Online serving endpoint is alive and ready to accept traffic
    READY = 'ready'

    # Asleep to conserve resources, due to manual update or inactivity timeout
    DORMANT = 'dormant'

    # We are still in progress to materialize data, provision resources,
    # or starting up server replicas.
    IN_PROGRESS = 'in_progress'

    # Failed to launch online serving endpoint, likely due to reasons such as
    # using an old model incompatible with online serving, insufficient
    # resources to launch too many replicas, etc.
    FAILED = 'failed'


@dataclass
class OnlineServingStatus:
    status_code: OnlineServingStatusCode

    # Most recently updated timestamp of current status.
    last_updated_at: datetime

    # Current stage while status_code is IN_PROGRESS.
    stage: Optional[str] = None
    # Message if status_code is FAILED.
    failure_message: Optional[str] = None


@dataclass
class OnlineServingUpdate:
    """
    Information/status about an update (PATCH) operation on an existing
    online serving endpoint.
    """
    prev_config: OnlineServingEndpointRequest
    target_config: OnlineServingEndpointRequest

    update_started_at: datetime
    update_status: OnlineServingStatus


@dataclass
class OnlineServingEndpointResource:
    id: str

    # Endpoint url would formatted as "<kumo cloud hostname>/gira/{id}"
    # where <kumo cloud hostname> is typical the your Kumo cloud web url such
    # as "https://<customer_id>.kumoai.cloud"
    endpoint_url: str

    config: OnlineServingEndpointRequest

    # Timestamp of when this endpoint resoruce was create.
    launched_at: datetime

    # Current status. The endpoint_url will be ready to serve traffic only if
    # status.status_code is READY
    status: OnlineServingStatus

    # The info/status about the most recent UPDATE operation on this endpoint,
    # if any.  Note that if the last update status is READY,
    # `update.target_config` would be identical to the `config` field,
    # otherwise `update.prev_config` would be identical to the `config` field.
    update: Optional[OnlineServingUpdate] = None


NodeId = Union[int, float, str]
TimestampNanos = int
NewNodeList = List[Tuple[NodeId, TimestampNanos]]
NewEdgeList = List[Tuple[NodeId, NodeId]]
# Row-oriented pandas dataframe for node features.
FeaturesDataframe = List[Dict[str, Any]]


@dataclass
class RealtimeFeatures:
    # TODO(siyang): the fields are used for testing now.

    # We are using List[Tuple] instead of Dict because TableInfo
    # as a key is not well supported when we serialize the object
    # before sending the request.
    # New nodes are represented as list of pairs of table info and
    # a list of tuples (node idx, time).
    # This field is only needed when the the node types have
    # timestamp columns.
    new_nodes: Optional[Dict[TableName, NewNodeList]] = None

    # New edges are represented as list of pairs of table info
    # and a list of tuples (node idx of table 1, node idx of table 2).
    new_edges: Optional[List[Tuple[Edge, NewEdgeList]]] = None

    # New features are represented as dict of table and row-oriented dataframe
    # node features in that table, where each row represents features of node.
    new_features: Optional[Dict[TableName, FeaturesDataframe]] = None


@dataclass
class OnlinePredictionRequest:
    fkey: NodeId
    time: Optional[TimestampNanos] = None
    realtime_features: Optional[RealtimeFeatures] = None


@dataclass
class BinaryClassificationResult:
    """Class which represents an online binary classification result.

    Below are an example batch prediction DataFrame for a single entity, and
    the corresponding BinaryClassificationResult.

    ENTITY  TARGET_PRED  False_PROB  True_PROB
       290        False    0.638515   0.361485

    BinaryClassificationResult(pred=False,
                               true_prob=0.3614853620529175,
                               type=binary_classification)

    Attributes:
        pred (bool): The predicted class.
        true_prob (float): The probability of the positive class.
        type (Literal[TaskType.BINARY_CLASSIFICATION]): Tag which is needed
            because this class is a member of the OnlinePredictionResult tagged
            union. A default value is assigned, and it should not be changed.
    """
    pred: bool
    true_prob: float
    type: Literal[
        TaskType.BINARY_CLASSIFICATION] = TaskType.BINARY_CLASSIFICATION

    @property
    def false_prob(self) -> float:
        """float: The probability of the negative class."""
        return 1 - self.true_prob


class _Config:
    smart_union = True


_config = ConfigDict() if WITH_PYDANTIC_V2 else _Config

# any Dtype supported by Stype.categorical except bool
NoBoolClassLabelType = TypeVar('NoBoolClassLabelType', int, float, str)


@dataclass(config=_config)
class MulticlassClassificationResult(Generic[NoBoolClassLabelType]):
    """Class which represents an online multiclass classification result.

    This class is generic over NoBoolClassLabelType, which is the type of the
    labels. NoBoolClassLabelType is constrained to int, float, or str, i.e. any
    Dtype under Stype.categorical with the exception of bool, which would make
    the task a binary classification task rather than a multiclass
    classification task.

    Below are an example batch prediction DataFrame for a single entity, and
    the corresponding MulticlassClassificationResult.

    ENTITY  PREDICTED  CLASS     SCORE
       290       True    dog  0.393810
       290      False    cat  0.261890
       290      False  mouse  0.177053
       290      False   bird  0.124285
       290      False  snake  0.042962

    MulticlassClassificationResult(
        pred=[('dog', 0.39381009340286255, True),
              ('cat', 0.26188984513282776, False),
              ('mouse', 0.17705298960208893, False),
              ('bird', 0.12428521364927292, False),
              ('snake', 0.042961835861206055, False)],
        type=multiclass_classification)

    Attributes:
        pred (List[Tuple[NoBoolClassLabelType, float, bool]]): All of the
            possible labels for the task, plus their scores, plus boolean flags
            which indicate that the label at the front of the list is predicted
            and the rest are not.
        type (Literal[TaskType.MULTICLASS_CLASSIFICATION]): Tag which is needed
            because this class is a member of the OnlinePredictionResult tagged
            union. A default value is assigned, and it should not be changed.
    """
    pred: List[Tuple[NoBoolClassLabelType, float, bool]]
    type: Literal[
        TaskType.
        MULTICLASS_CLASSIFICATION] = TaskType.MULTICLASS_CLASSIFICATION


# any Dtype supported by Stype.categorical
ClassLabelType = TypeVar('ClassLabelType', bool, int, float, str)


@dataclass(config=_config)
class MultilabelClassificationResult(Generic[ClassLabelType]):
    """Class which represents an online multilabel classification result.

    This class is generic over ClassLabelType, which is the type of the labels.
    ClassLabelType is constrained to bool, int, float, or str, i.e. any Dtype
    under Stype.categorical.

    Below are an example batch prediction DataFrame for a single entity, and
    the corresponding MultilabelClassificationResult.

    ENTITY  PREDICTED   CLASS     SCORE
       290       True   sweet  0.710820
       290       True    sour  0.608831
       290       True   salty  0.544676
       290      False   spicy  0.488941
       290      False  bitter  0.372759
       290      False   umami  0.362013

    MultilabelClassificationResult(
        pred=[('sweet', 0.7108199596405029, True),
              ('sour', 0.6088314056396484, True),
              ('salty', 0.5446761846542358, True),
              ('spicy', 0.488941490650177, False),
              ('bitter', 0.3727594316005707, False),
              ('umami', 0.36201322078704834, False)],
        type=multilabel_classification)

    Attributes:
        pred (List[Tuple[ClassLabelType, float, bool]]): All of the possible
            labels for the task, plus their scores, plus boolean flags which
            indicate whether each is predicted or not.
        type (Literal[TaskType.MULTILABEL_CLASSIFICATION]): Tag which is needed
            because this class is a member of the OnlinePredictionResult tagged
            union. A default value is assigned, and it should not be changed.
    """
    pred: List[Tuple[ClassLabelType, float, bool]]
    type: Literal[
        TaskType.
        MULTILABEL_CLASSIFICATION] = TaskType.MULTILABEL_CLASSIFICATION


@dataclass
class RegressionResult:
    """Class which represents an online regression result.

    Below are an example batch prediction DataFrame for a single entity, and
    the corresponding RegressionResult.

    ENTITY  TARGET_PRED
       290     0.569136

    RegressionResult(pred=0.5691359639167786, type=regression)

    Attributes:
        pred (float): The predicted value.
        type (Literal[TaskType.REGRESSION]): Tag which is needed because this
            class is a member of the OnlinePredictionResult tagged union. A
            default value is assigned, and it should not be changed.
    """
    pred: float
    type: Literal[TaskType.REGRESSION] = TaskType.REGRESSION


# any Dtype supported by Stype.ID
NodeIdType = TypeVar('NodeIdType', int, float, str)


@dataclass(config=_config)
class LinkPredictionResult(Generic[NodeIdType]):
    """Class which represents an online link prediction result.

    This class is generic over NodeIdType, which is the type of the RHS node
    IDs. NodeIdType is constrained to int, float, or str, i.e. any Dtype under
    Stype.ID.

    Below are an example batch prediction DataFrame for a single entity, and
    the corresponding LinkPredictionResult.

    ENTITY  CLASS      SCORE
      1806   3068  12.540409
      1806   2187  11.612039

    LinkPredictionResult(pred=[(3068, 12.540409088134766),
                               (2187, 11.612038612365723)],
                         type=static_link_prediction)

    Attributes:
        pred (List[Tuple[NodeIdType, float]]): The RHS node IDs and scores of
            the predicted links.
        type (Literal[TaskType.STATIC_LINK_PREDICTION,
                      TaskType.TEMPORAL_LINK_PREDICTION]): Tag which is needed
            because this class is a member of the OnlinePredictionResult tagged
            union. Should be initialized to either
            TaskType.STATIC_LINK_PREDICTION or
            TaskType.TEMPORAL_LINK_PREDICTION.
    """
    pred: List[Tuple[NodeIdType, float]]
    type: Literal[TaskType.STATIC_LINK_PREDICTION,
                  TaskType.TEMPORAL_LINK_PREDICTION]


OnlinePredictionResult = Union[BinaryClassificationResult,
                               MulticlassClassificationResult,
                               MultilabelClassificationResult,
                               RegressionResult, LinkPredictionResult]


@dataclass
class OnlinePredictionResponse:
    result: OnlinePredictionResult = Field(discriminator='type')
