from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence


class RecurrentBaseline(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, name: str):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.name = name

    def forward(
        self, x: Union[Tensor, PackedSequence], state: List[Tensor]
    ) -> Tuple[Union[Tensor, PackedSequence]]:
        """
        Forward pass through the recurrent network
        :param x: input tensor for the RNN with shape (seq, batch, feature) which can be a packed sequence
        :param state: list of state tensors with sizes matching those returned by initial_state + the batch dimension
            if initial_state returns [tensor(2, 10)], then with batch 4, state will be [tensor(4, 2, 10)]
            sometimes this batching is undesirable for torch RNNs so you must reshape them accordingly
        :returns:
            output: the output used for network heads.
            rnn_state: a list of next hidden states with shape equivalent to the state argument.
        """
        raise NotImplementedError

    def initial_state(self) -> List[Tensor]:
        """
        Get the initial state for the recurrent network
        :returns: a list of inputs for the RNN without the batch dimension
        """
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def custom_loss(self):
        return 0


class GRUBaseline(RecurrentBaseline):
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super().__init__(input_size, hidden_size, num_layers, "gru")
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, **kwargs)

    def initial_state(self):
        h_0 = torch.zeros(self.num_layers, self.hidden_size)
        return [h_0]

    def forward(self, x, state):
        # x shape tensor(seq, batch, feature) or PackedSequence
        assert isinstance(x, torch.Tensor) or isinstance(x, nn.utils.rnn.PackedSequence)
        # state shape [tensor(batch, layers, feature)]
        assert isinstance(state, list)
        if len(state) == 1:
            h_0 = state[0]
            assert isinstance(h_0, torch.Tensor)
            assert len(h_0.size()) == 3
            assert h_0.size(1) == self.num_layers
            # permute to shape tensor(layers, batch, feature)
            h_0 = h_0.to(self.device).permute(1, 0, 2)
        elif len(state) == 0:
            h_0 = None
        else:
            raise NotImplementedError
        output, h_n = self.rnn(x.to(self.device), h_0)
        # change back to shape [tensor(batch, layers, feature)]
        rnn_state = [h_n.permute(1, 0, 2)]
        return output, rnn_state


class LSTMBaseline(RecurrentBaseline):
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super().__init__(input_size, hidden_size, num_layers, "lstm")
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, **kwargs)

    def initial_state(self):
        h_0 = torch.zeros(self.num_layers, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, self.hidden_size)
        return [h_0, c_0]

    def forward(self, x, state):
        # x shape tensor(seq, batch, feature) or PackedSequence
        assert isinstance(x, torch.Tensor) or isinstance(x, nn.utils.rnn.PackedSequence)
        # state shape [tensor(batch, layers, feature)]
        assert isinstance(state, list)
        if len(state) == 2:
            h_0, c_0 = state
            assert isinstance(h_0, torch.Tensor)
            assert isinstance(c_0, torch.Tensor)
            assert len(h_0.size()) == 3
            assert len(c_0.size()) == 3
            assert h_0.size(1) == self.num_layers
            assert c_0.size(1) == self.num_layers
            # permute to shape tensor(layers, batch, feature)
            h_0 = h_0.to(self.device).permute(1, 0, 2)
            c_0 = c_0.to(self.device).permute(1, 0, 2)
        elif len(state) == 0:
            h_0, c_0 = None, None
        else:
            raise NotImplementedError
        output, (h_n, c_n) = self.rnn(x.to(self.device), (h_0, c_0))
        # change back to shape [tensor(batch, layers, feature)]
        rnn_state = [h_n.permute(1, 0, 2), c_n.permute(1, 0, 2)]
        return output, rnn_state
