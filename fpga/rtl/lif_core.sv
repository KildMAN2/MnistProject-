// -----------------------------------------------------------------------------
// lif_core : bit-exact, combinational LIF update (RTL milestone 2).
//
// Reference: Int16SNN._lif_step() in software/quantization/quantization.py
//   previous_reset = membrane > threshold
//   decayed        = round_ties_to_even(19*membrane/20)         (decay_19_20)
//   pre_spike      = decayed + current - previous_reset*threshold
//   spike          = pre_spike > threshold                       (STRICT)
//   membrane_new   = pre_spike - (spike - previous_reset)*threshold
//
// Hardware form (algebraically identical, see docs):
//   base         = decayed + current
//   pre_spike    = base - (prev_reset ? thr : 0)     <- prev_reset stays here
//   spike_out    = pre_spike > thr
//   membrane_out = base - (spike_out ? thr : 0)
//
// All arithmetic is signed MEM_WIDTH. The width analysis (docs) shows that
// decayed/base/pre_spike/membrane_out fit in MEM_WIDTH for every reachable
// membrane/current pair (L1: 38/35, L2: 36/33); the stored membrane and the
// current must stay inside that analysed range (inputs outside it can wrap).
// No state: the membrane storage belongs to higher-level logic.
// -----------------------------------------------------------------------------
module lif_core #(
    parameter int     MEM_WIDTH = 38,
    parameter int     CUR_WIDTH = 35,
    parameter longint THRESHOLD = 37718198
) (
    input  logic signed [MEM_WIDTH-1:0] membrane_in,
    input  logic signed [CUR_WIDTH-1:0] current_in,
    output logic signed [MEM_WIDTH-1:0] membrane_out,
    output logic                        spike_out
);

  // ---- elaboration checks ---------------------------------------------------
  initial begin
    if (CUR_WIDTH > MEM_WIDTH)
      $fatal(1, "lif_core: CUR_WIDTH (%0d) must be <= MEM_WIDTH (%0d)", CUR_WIDTH, MEM_WIDTH);
    if (THRESHOLD <= 0 || THRESHOLD >= (64'sd1 <<< (MEM_WIDTH - 1)))
      $fatal(1, "lif_core: THRESHOLD %0d must be positive and fit in signed %0d bits",
             THRESHOLD, MEM_WIDTH);
  end

  // ---- threshold as an explicit signed MEM_WIDTH constant -------------------
  localparam logic signed [MEM_WIDTH-1:0] THR = THRESHOLD;

  // ---- explicit sign extension of the current -------------------------------
  logic signed [MEM_WIDTH-1:0] cur_ext;
  // current_in is signed, so plain assignment sign-extends (CUR_WIDTH <= MEM_WIDTH).
  assign cur_ext = current_in;

  // ---- previous reset: compares the STORED membrane (not decayed/base) ------
  logic prev_reset;
  assign prev_reset = (membrane_in > THR);

  // ---- exact decay (verified block, not duplicated) -------------------------
  logic signed [MEM_WIDTH-1:0] decayed;
  decay_19_20 #(.MEM_WIDTH(MEM_WIDTH)) u_decay (
      .membrane_in(membrane_in),
      .decayed_out(decayed)
  );

  // ---- base, pre_spike, spike, membrane update -------------------------------
  logic signed [MEM_WIDTH-1:0] base, pre_spike;
  assign base      = decayed + cur_ext;
  assign pre_spike = prev_reset ? (base - THR) : base;
  assign spike_out = (pre_spike > THR);
  assign membrane_out = spike_out ? (base - THR) : base;

endmodule
