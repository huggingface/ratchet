use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    up_proj: Linear,
    down_proj: Linear,
}

//class Phi3MLP(nn.Module):
//    def __init__(self, config):
//        super().__init__()
//
//        self.config = config
//        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
//        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
//
//        self.activation_fn = ACT2FN[config.hidden_act]
//
//    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
//        up_states = self.gate_up_proj(hidden_states)
//
//        gate, up_states = up_states.chunk(2, dim=-1)
//        up_states = up_states * self.activation_fn(gate)
//
//        return self.down_proj(up_states)

impl Module for MLP {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<ratchet::Tensor> {
        let up_states = self.up_proj.schedule(input)?;
        let [x, y, z]: [usize; 3] = up_states.shape().try_into()?;
        let gate = up_states.clone().slice(&[0..x, 0..y, 0..z / 2])?;
        let up_states = up_states.clone().slice(&[0..x, 0..y, z / 2..z])?;
        let up_states = up_states.mul(gate.silu()?)?;
        self.down_proj.schedule(up_states)
    }
}
