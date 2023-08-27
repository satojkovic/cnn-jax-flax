from clu import metrics
from flax.training import train_state
from flax import struct
import optax


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
