"""LSTM model, param-count helpers and the gate-inspection pass (manual nn.LSTMCell unroll)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=None, num_layers=None, dropout=None):
        super().__init__()
        hidden = hidden or C.HIDDEN; num_layers = num_layers or C.NUM_LAYERS
        dropout = C.DROPOUT if dropout is None else dropout
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)                        # [exposure logit, next-day return (standardised)]
        with torch.no_grad():
            self.head.bias[0] = C.W_INIT_LOGIT; self.head.bias[1] = 0.0   # start near Buy & Hold (exposure ~0.9)
            # forget-gate bias init (gate order i,f,g,o)
            for name, p in self.lstm.named_parameters():
                if name.startswith("bias_ih"):
                    h = p.numel() // 4; p[h:2 * h].fill_(C.FORGET_BIAS_INIT)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])                       # (batch, 2); last timestep only


class RNNReference(nn.Module):
    """Structure of the Simple-RNN counterpart; used ONLY to count parameters / match capacity."""
    def __init__(self, n_features):
        super().__init__()
        self.rnn = nn.RNN(n_features, C.HIDDEN, C.NUM_LAYERS, batch_first=True, dropout=C.DROPOUT)
        self.head = nn.Linear(C.HIDDEN, 1)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def match_hidden(n_features):
    """Hidden size whose 2-layer LSTM param count is closest to the Simple RNN's."""
    target = count_params(RNNReference(n_features))
    best = min(range(8, C.HIDDEN + 1), key=lambda h: abs(count_params(LSTMModel(n_features, h)) - target))
    return best, target


@torch.no_grad()
def extract_gates(model, Xw, device, chunk=512):
    """Unroll the trained nn.LSTM manually with nn.LSTMCell (weights copied from the model).
    Gate order in PyTorch is (i, f, g, o). Returns, per decision sample:
      f_mean_t, i_mean_t : (n, W, L)  gate activations averaged over hidden units, every timestep
      last_f, last_i, last_c : (n, L, H) values at the final timestep (inputs to the prediction)
    Verified against nn.LSTM's own output."""
    model.eval()
    lstm = model.lstm; L, H, W = lstm.num_layers, lstm.hidden_size, Xw.shape[1]
    cells = []
    for l in range(L):
        cell = nn.LSTMCell(lstm.input_size if l == 0 else H, H).to(device)
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(cell, n).copy_(getattr(lstm, f"{n}_l{l}"))
        cells.append(cell.eval())
    n = len(Xw)
    out = dict(f_mean_t=np.zeros((n, W, L), np.float32), i_mean_t=np.zeros((n, W, L), np.float32),
               last_f=np.zeros((n, L, H), np.float32), last_i=np.zeros((n, L, H), np.float32),
               last_c=np.zeros((n, L, H), np.float32))
    for s in range(0, n, chunk):
        x = torch.from_numpy(Xw[s:s + chunk]).to(device); b = x.shape[0]
        h = [torch.zeros(b, H, device=device) for _ in range(L)]
        c = [torch.zeros(b, H, device=device) for _ in range(L)]
        for t in range(W):
            inp = x[:, t]
            for l, cell in enumerate(cells):
                gates = F.linear(inp, cell.weight_ih, cell.bias_ih) + F.linear(h[l], cell.weight_hh, cell.bias_hh)
                gi, gf, _, _ = gates.chunk(4, dim=1)
                gi, gf = torch.sigmoid(gi), torch.sigmoid(gf)
                h[l], c[l] = cell(inp, (h[l], c[l]))          # official cell update (same weights)
                inp = h[l]
                out["f_mean_t"][s:s + b, t, l] = gf.mean(1).cpu().numpy()
                out["i_mean_t"][s:s + b, t, l] = gi.mean(1).cpu().numpy()
                if t == W - 1:
                    out["last_f"][s:s + b, l] = gf.cpu().numpy(); out["last_i"][s:s + b, l] = gi.cpu().numpy()
                    out["last_c"][s:s + b, l] = c[l].cpu().numpy()
        if s == 0:                                            # equivalence check vs nn.LSTM
            ref, _ = lstm(x)
            assert torch.allclose(ref[:, -1], h[-1], atol=1e-4), "manual unroll != nn.LSTM output"
    return out
