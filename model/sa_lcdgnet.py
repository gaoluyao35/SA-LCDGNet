"""SA-LCDGNet: State-Aware Lag-Conditioned Dynamic Graph Network.

Traffic-flow forecasting model designed for integration with BasicTS.
The temporal convolution and graph propagation building blocks are adapted
from the Graph WaveNet baseline in BasicTS; the state-aware, lag-conditioned
and memory-enhanced modeling components implement SA-LCDGNet.

Retain applicable upstream copyright and license notices when redistributing.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class GraphNeighborhoodAggregation(nn.Module):
    """Aggregate node features using static or batch-dependent adjacency."""

    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        if A.dim() == 2:
            x = torch.einsum('bcnt,nm->bcmt', (x, A))
        elif A.dim() == 3:
            x = torch.einsum('bcnt,bnm->bcmt', (x, A))
        else:
            raise ValueError(f"Adjacency must be 2D or 3D, got {A.dim()}D")
        return x.contiguous()


class PointwiseLinear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class MultiGraphPropagation(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = GraphNeighborhoodAggregation()
        c_in = (order * support_len + 1) * c_in
        self.mlp = PointwiseLinear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            a = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



class TrafficStateTransitionModule(nn.Module):
    """Traffic State Transition Module (TSTM).

    Uses first- and second-order temporal differences together with a learned
    transition prior to infer node-level traffic-state summaries for dynamic
    graph construction. The TCN backbone features remain unchanged.

    Args:
        x: [B, C, N, T] temporal feature tensor.
    Returns:
        state_summary_last: [B, N, state_dim].
    """

    def __init__(
        self,
        channels,
        num_states=4,
        state_dim=16,
        hidden_dim=None,
        dropout=0.1,
        transition_mix_init=0.03,
    ):
        super().__init__()
        self.channels = channels
        self.num_states = num_states
        self.state_dim = state_dim
        self.eps = 1e-8

        if hidden_dim is None:
            hidden_dim = channels

        # First- and second-order temporal variations are the core state cues.
        self.delta_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=True)
        self.accel_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=True)

        self.state_mlp = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states),
        )

        self.state_emb = nn.Parameter(torch.empty(num_states, state_dim))
        nn.init.xavier_uniform_(self.state_emb)

        # Learnable transition prior among traffic states.
        # It is initialized close to an identity transition, which is consistent
        # with short-term traffic-state continuity.
        self.transition_logits = nn.Parameter(torch.eye(num_states))

        transition_mix_init = 0.03 if transition_mix_init is None else transition_mix_init
        transition_mix_init = min(max(float(transition_mix_init), 1e-4), 1.0 - 1e-4)
        self.transition_mix_logit = nn.Parameter(
            torch.tensor(
                math.log(transition_mix_init / (1.0 - transition_mix_init)),
                dtype=torch.float32
            )
        )

        # Optional outputs for interpreting the learned traffic states.
        self.last_state_probs = None
        self.last_state_summary = None
        self.last_transition_matrix = None
        self.last_transition_mix = None

    @staticmethod
    def _temporal_diff(x):
        # x: [B, C, N, T]
        dx = torch.zeros_like(x)
        dx[..., 1:] = x[..., 1:] - x[..., :-1]
        return dx

    def forward(self, x):
        # x: [B, C, N, T]
        B, C, N, T = x.shape

        delta = self._temporal_diff(x)
        accel = self._temporal_diff(delta)

        delta_feat = self.delta_proj(delta)
        accel_feat = self.accel_proj(accel)

        # [B, N, T, 3C]
        state_in = torch.cat(
            [
                x.permute(0, 2, 3, 1),
                delta_feat.permute(0, 2, 3, 1),
                accel_feat.permute(0, 2, 3, 1),
            ],
            dim=-1,
        ).contiguous()

        local_logits = self.state_mlp(state_in)  # [B, N, T, S]

        # Recursively combine local state evidence with the transition prior.
        trans_mat = torch.softmax(self.transition_logits, dim=-1)  # [S, S]
        mix = torch.sigmoid(self.transition_mix_logit)

        state_probs_list = []
        prev_prob = None
        for t in range(T):
            logits_t = local_logits[:, :, t, :]  # [B, N, S]

            if prev_prob is None:
                prob_t = torch.softmax(logits_t, dim=-1)
            else:
                transition_prior = torch.einsum('bns,sk->bnk', prev_prob, trans_mat)
                logits_fused = logits_t + mix * torch.log(transition_prior + self.eps)
                prob_t = torch.softmax(logits_fused, dim=-1)

            state_probs_list.append(prob_t)
            prev_prob = prob_t

        state_probs = torch.stack(state_probs_list, dim=2)  # [B, N, T, S]

        state_ctx = torch.einsum('bnts,sd->bntd', state_probs, self.state_emb)
        state_summary_last = state_ctx[:, :, -1, :]  # [B, N, state_dim]

        self.last_state_probs = state_probs.detach()
        self.last_state_summary = state_summary_last.detach()
        self.last_transition_matrix = trans_mat.detach()
        self.last_transition_mix = mix.detach()

        return state_summary_last


class LagConditionedDynamicGraph(nn.Module):
    """Lag-Conditioned Dynamic Graph (LCDG).

    Learns node-specific weights over candidate lag steps, constructs lag
    summaries, and combines current, lag and traffic-state representations
    to obtain an input-dependent directed adjacency matrix.

    Args:
        x: [B, C, N, T] temporal feature tensor.
        state_summary: [B, N, state_dim] or None.
    Returns:
        Dynamic adjacency, lag weights, and intermediate representations.
    """

    def __init__(
        self,
        num_nodes,
        channels,
        lag_list=(0, 1, 2, 3),
        dyn_graph_dim=16,
        state_dim=16,
        dropout=0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.lag_list = list(lag_list)
        self.num_lags = len(self.lag_list)
        self.state_dim = state_dim
        self.eps = 1e-8

        self.lag_emb = nn.Parameter(torch.empty(self.num_lags, dyn_graph_dim))
        nn.init.xavier_uniform_(self.lag_emb)

        # Direction-specific node representations for dynamic adjacency construction.
        self.start_emb = nn.Parameter(torch.empty(num_nodes, dyn_graph_dim))
        self.end_emb = nn.Parameter(torch.empty(num_nodes, dyn_graph_dim))
        nn.init.xavier_uniform_(self.start_emb)
        nn.init.xavier_uniform_(self.end_emb)

        self.route_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, self.num_lags),
        )

        qk_in_dim = channels * 2 + state_dim
        self.node_q_proj = nn.Linear(qk_in_dim, dyn_graph_dim)
        self.node_k_proj = nn.Linear(qk_in_dim, dyn_graph_dim)

        self.dropout = nn.Dropout(dropout)

        self.last_alpha = None
        self.last_raw_adj = None
        self.last_edge_rel = None
        self.last_dyn_adj = None

    def forward(self, x, state_summary):
        # x: [B, C, N, T]
        B, C, N, T = x.shape

        curr = x[:, :, :, -1].permute(0, 2, 1).contiguous()  # [B, N, C]

        lag_feats = []
        for lag in self.lag_list:
            idx = max(T - 1 - lag, 0)
            lag_feats.append(x[:, :, :, idx].permute(0, 2, 1).contiguous())
        lag_bank = torch.stack(lag_feats, dim=-1)             # [B, N, C, L]

        alpha = torch.softmax(self.route_mlp(curr), dim=-1)   # [B, N, L]
        lag_summary = (lag_bank * alpha.unsqueeze(2)).sum(dim=-1)  # [B, N, C]
        lag_ctx = torch.einsum('bnl,ld->bnd', alpha, self.lag_emb) # [B, N, Dg]

        node_state = torch.cat([curr, lag_summary, state_summary], dim=-1)

        q = self.node_q_proj(node_state) + self.start_emb.unsqueeze(0) + lag_ctx
        k = self.node_k_proj(node_state) + self.end_emb.unsqueeze(0) + lag_ctx

        raw_logits = torch.einsum('bnd,bmd->bnm', q, k) / math.sqrt(q.size(-1))
        raw_logits = F.relu(raw_logits)
        dyn_adj = torch.softmax(raw_logits, dim=-1)
        dyn_adj = self.dropout(dyn_adj)

        # The all-one reliability matrix is retained for diagnostic compatibility.
        edge_rel = torch.ones_like(dyn_adj)

        self.last_alpha = alpha.detach()
        self.last_raw_adj = dyn_adj.detach()
        self.last_edge_rel = edge_rel.detach()
        self.last_dyn_adj = dyn_adj.detach()

        aux = {
            "curr": curr,
            "lag_summary": lag_summary,
            "lag_ctx": lag_ctx,
        }
        return dyn_adj, alpha, dyn_adj, edge_rel, aux


class MemoryEnhancementModule(nn.Module):
    """Memory enhancement for stable spatial relations.

    Retrieves learned node-relation patterns with a lag-conditioned query,
    then fuses the memory adjacency and LCDG adjacency using a learned gate.
    """

    def __init__(
        self,
        channels,
        num_nodes,
        dyn_graph_dim=16,
        beta_init=0.03,
        dropout=0.0,
    ):
        super().__init__()
        self.eps = 1e-8

        self.memory = nn.Parameter(torch.empty(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)

        self.query_proj = nn.Linear(channels * 2 + dyn_graph_dim, channels)

        beta_init = min(max(beta_init, 1e-4), 1 - 1e-4)
        self.beta_logit = nn.Parameter(
            torch.tensor(math.log(beta_init / (1.0 - beta_init)), dtype=torch.float32)
        )

        self.dropout = nn.Dropout(dropout)

        self.last_memory_adj = None
        self.last_beta = None
        self.last_fused_adj = None

    def forward(self, curr, lag_summary, lag_ctx, lcdg_adj):
        mem_query = self.query_proj(torch.cat([curr, lag_summary, lag_ctx], dim=-1))  # [B,N,C]
        mem_logits = torch.einsum("bnc,cm->bnm", mem_query, self.memory).contiguous() / math.sqrt(mem_query.size(-1))
        memory_adj = torch.softmax(F.relu(mem_logits), dim=-1)

        beta = torch.sigmoid(self.beta_logit)
        fused_adj = (1.0 - beta) * lcdg_adj + beta * memory_adj
        fused_adj = fused_adj / fused_adj.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        fused_adj = self.dropout(fused_adj)

        self.last_memory_adj = memory_adj.detach()
        self.last_beta = beta.detach()
        self.last_fused_adj = fused_adj.detach()
        return fused_adj


class SALCDGNet(nn.Module):
    """Full SA-LCDGNet for short-term traffic-flow forecasting.

    Always uses temporal embeddings, TSTM, LCDG, memory enhancement,
    static road-network graphs, an adaptive graph, and multi-graph propagation.
    TSTM acts on the first ``tstm_num_layers`` layers; memory enhancement acts
    on the last ``memory_num_layers`` layers (these are architectural settings,
    not ablation controls).

    Input: history_data [B, T_h, N, 3] (flow, time of day, day of week).
    Output: [B, out_dim, N, 1], for BasicTS compatibility.
    """

    def __init__(
        self,
        num_nodes,
        supports,
        dropout=0.3,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        lag_list=(0, 1, 2, 3),
        dyn_graph_dim=16,
        steps_per_day=288,
        memory_num_layers=2,
        memory_beta_init=0.03,
        memory_dropout=0.0,
        gate_init_bias=-2.2,
        temporal_emb_init=0.02,
        tstm_num_layers=2,
        tstm_num_states=4,
        tstm_state_dim=16,
        tstm_hidden_dim=32,
        tstm_dropout=0.1,
        tstm_transition_mix_init=0.05,
    ):
        super().__init__()
        if not supports:
            raise ValueError("Full SA-LCDGNet requires static road-network support matrices.")
        if not (1 <= tstm_num_layers <= blocks * layers):
            raise ValueError("tstm_num_layers must be between 1 and blocks * layers.")
        if not (1 <= memory_num_layers <= blocks * layers):
            raise ValueError("memory_num_layers must be between 1 and blocks * layers.")

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.lag_list = list(lag_list)
        self.steps_per_day = steps_per_day
        self.tstm_state_dim = tstm_state_dim

        # Input embedding uses flow, time-of-day, and day-of-week features.
        self.value_proj = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
        self.tod_emb = nn.Embedding(steps_per_day, residual_channels)
        nn.init.normal_(self.tod_emb.weight, mean=0.0, std=temporal_emb_init)
        self.tod_gate = nn.Parameter(torch.full((1, residual_channels, 1, 1), gate_init_bias))
        self.dow_emb = nn.Embedding(7, residual_channels)
        nn.init.normal_(self.dow_emb.weight, mean=0.0, std=temporal_emb_init)
        self.dow_gate = nn.Parameter(torch.full((1, residual_channels, 1, 1), gate_init_bias))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # Retained for loading checkpoints and preserving the upstream RNG
        # initialization sequence. The full model always uses self.gconv.
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.dynamic_graph_builders = nn.ModuleList()
        self.memory_enhancers = nn.ModuleList()
        self.tstm_blocks = nn.ModuleList()
        self.tstm_active_flags = []

        self.supports = list(supports)
        receptive_field = 1
        total_layers = blocks * layers
        self.memory_num_layers = memory_num_layers
        self.tstm_num_layers = tstm_num_layers

        # Two spatial supports are added to the supplied static supports:
        # one adaptive graph and one input-dependent LCDG graph.
        self.supports_len = len(self.supports) + 2
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)

        layer_idx = 0
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                tstm_active = layer_idx < self.tstm_num_layers
                self.tstm_active_flags.append(tstm_active)
                if tstm_active:
                    self.tstm_blocks.append(
                        TrafficStateTransitionModule(
                            channels=dilation_channels,
                            num_states=tstm_num_states,
                            state_dim=tstm_state_dim,
                            hidden_dim=tstm_hidden_dim,
                            dropout=tstm_dropout,
                            transition_mix_init=tstm_transition_mix_init,
                        )
                    )
                else:
                    self.tstm_blocks.append(nn.Identity())

                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(
                    MultiGraphPropagation(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.supports_len,
                        order=2,
                    )
                )
                self.dynamic_graph_builders.append(
                    LagConditionedDynamicGraph(
                        num_nodes=num_nodes,
                        channels=dilation_channels,
                        lag_list=self.lag_list,
                        dyn_graph_dim=dyn_graph_dim,
                        state_dim=tstm_state_dim,
                        dropout=dropout,
                    )
                )
                if layer_idx >= total_layers - self.memory_num_layers:
                    self.memory_enhancers.append(
                        MemoryEnhancementModule(
                            channels=dilation_channels,
                            num_nodes=num_nodes,
                            dyn_graph_dim=dyn_graph_dim,
                            beta_init=memory_beta_init,
                            dropout=memory_dropout,
                        )
                    )
                else:
                    self.memory_enhancers.append(None)

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                layer_idx += 1

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.receptive_field = receptive_field

        # Non-persistent diagnostic tensors, useful for interpreting graphs.
        self.last_adp = None
        self.last_dyn_adj = None
        self.last_dyn_raw_adj = None
        self.last_dyn_edge_rel = None
        self.last_lag_alpha = None
        self.last_memory_adj = None
        self.last_memory_beta = None
        self.last_fused_adj = None
        self.last_tstm_state_probs = None
        self.last_tstm_state_summary = None
        self.last_tstm_transition_matrix = None
        self.last_cached_state_summary = None

    def _extract_tod_dow(self, history_data):
        if history_data.size(-1) < 3:
            raise ValueError("Full SA-LCDGNet requires flow, TOD and DOW input features.")
        tod = history_data[..., 1]
        if tod.max() <= 1.5:
            tod_idx = torch.clamp((tod * self.steps_per_day).long(), 0, self.steps_per_day - 1)
        else:
            tod_idx = torch.clamp(tod.long(), 0, self.steps_per_day - 1)

        dow = history_data[..., 2]
        if dow.max() <= 1.5:
            dow_idx = torch.clamp((dow * 7).long(), 0, 6)
        else:
            dow_idx = torch.clamp(dow.long(), 0, 6)
        return tod_idx, dow_idx

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        tod_idx, dow_idx = self._extract_tod_dow(history_data)

        value = history_data[..., 0:1].transpose(1, 3).contiguous()
        x = self.value_proj(value)
        tod_feat = self.tod_emb(tod_idx).permute(0, 3, 2, 1).contiguous()
        x = x + torch.sigmoid(self.tod_gate) * tod_feat
        dow_feat = self.dow_emb(dow_idx).permute(0, 3, 2, 1).contiguous()
        x = x + torch.sigmoid(self.dow_gate) * dow_feat
        skip = 0

        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        self.last_adp = adp.detach()

        cached_state_summary = None
        for i in range(self.blocks * self.layers):
            residual = x

            # Gated dilated causal temporal convolution.
            dilation = self.filter_convs[i].dilation[1]
            kernel_size = self.filter_convs[i].kernel_size[1]
            pad_len = (kernel_size - 1) * dilation
            x_padded = F.pad(residual, (pad_len, 0, 0, 0))
            filter_out = torch.tanh(self.filter_convs[i](x_padded))
            gate_out = torch.sigmoid(self.gate_convs[i](x_padded))
            x_tcn = filter_out * gate_out

            s = self.skip_convs[i](x_tcn)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            # TSTM supplies state summaries; the backbone feature is x_tcn.
            state_summary = cached_state_summary
            if self.tstm_active_flags[i]:
                state_summary = self.tstm_blocks[i](x_tcn)
                cached_state_summary = state_summary
                self.last_tstm_state_probs = self.tstm_blocks[i].last_state_probs
                self.last_tstm_state_summary = self.tstm_blocks[i].last_state_summary
                self.last_tstm_transition_matrix = self.tstm_blocks[i].last_transition_matrix
                self.last_cached_state_summary = cached_state_summary.detach()

            support_list = list(self.supports)
            support_list.append(adp)
            dyn_builder = self.dynamic_graph_builders[i]
            dyn_adj, lag_alpha, raw_adj, edge_rel, aux = dyn_builder(
                x_tcn, state_summary=state_summary
            )
            if self.memory_enhancers[i] is not None:
                dyn_adj = self.memory_enhancers[i](
                    aux["curr"], aux["lag_summary"], aux["lag_ctx"], dyn_adj
                )
                self.last_memory_adj = self.memory_enhancers[i].last_memory_adj
                self.last_memory_beta = self.memory_enhancers[i].last_beta
                self.last_fused_adj = self.memory_enhancers[i].last_fused_adj
            support_list.append(dyn_adj)

            self.last_dyn_adj = dyn_adj.detach()
            self.last_dyn_raw_adj = raw_adj.detach()
            self.last_dyn_edge_rel = edge_rel.detach()
            self.last_lag_alpha = lag_alpha.detach()

            x = self.gconv[i](x_tcn, support_list)
            x = x + residual
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[:, :, :, -1:]
