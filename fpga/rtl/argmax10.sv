// -----------------------------------------------------------------------------
// argmax10 : combinational argmax over the ten output spike counts
// (RTL milestone 11).
//
//   predicted_digit = index of the maximum count; on ties the LOWEST index wins,
//   exactly like np.argmax / torch.argmax and the golden Int16SNN classification.
//
// Reference structure: a sequential scan in increasing index order, i.e. a
// combinational chain of 9 comparators (counts[1..9] against the running best).
// The comparison is strictly ">": an equal count never replaces the earlier
// (lower) index. ">=" would pick the highest index on ties and is WRONG.
//
// Purely combinational, no clock and no register. The result follows the inputs
// immediately, so the future controller must only use predicted_digit after all
// NUM_STEPS count pulses have been applied to the counters.
// -----------------------------------------------------------------------------
module argmax10
  import snn_params_pkg::*;
(
    input  logic [SPIKE_COUNT_WIDTH-1:0] counts [0:OUTPUT_SIZE-1],

    output logic [3:0]                   predicted_digit,
    output logic [SPIKE_COUNT_WIDTH-1:0] max_count
);

  always_comb begin
    max_count       = counts[0];
    predicted_digit = 4'd0;
    for (int k = 1; k < OUTPUT_SIZE; k++) begin
      if (counts[k] > max_count) begin
        max_count       = counts[k];
        predicted_digit = 4'(k);
      end
    end
  end

endmodule
