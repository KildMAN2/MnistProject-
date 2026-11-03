// -----------------------------------------------------------------------------
// output_spike_counters : ten parallel 5-bit output spike counters
// (RTL milestone 10).
//
// counts[k] = number of time steps in which output neuron k spiked.
//
// Interface
//   clk, rst        clock, synchronous active-high reset (clears all counts)
//   clear_counts    pulse: all counts <= 0 at the next clock edge
//   count_enable    ONE-clock pulse per completed time step: every counts[k] with
//                   output_spikes[k] == 1 is incremented by exactly one
//   output_spikes   spike vector of the step (level input; only sampled on a
//                   clock edge where count_enable is high)
//   counts[0:9]     unsigned SPIKE_COUNT_WIDTH (5) bit counts
//
// Priority: reset > clear_counts > count_enable. clear_counts and count_enable on
// the same edge give all zeros (no increment).
// This block knows nothing about step_done: the future controller converts the
// Layer-2 completion event into a single count_enable pulse. A spike level that
// stays high while count_enable is low changes nothing.
// No saturation: a valid inference issues exactly NUM_STEPS = 25 pulses, so the
// maximum count is 25 < 2^5.
// -----------------------------------------------------------------------------
module output_spike_counters
  import snn_params_pkg::*;
(
    input  logic                          clk,
    input  logic                          rst,

    input  logic                          clear_counts,
    input  logic                          count_enable,
    input  logic [OUTPUT_SIZE-1:0]        output_spikes,

    output logic [SPIKE_COUNT_WIDTH-1:0]  counts [0:OUTPUT_SIZE-1]
);

  always_ff @(posedge clk) begin
    if (rst || clear_counts) begin
      for (int k = 0; k < OUTPUT_SIZE; k++) counts[k] <= '0;
    end else if (count_enable) begin
      for (int k = 0; k < OUTPUT_SIZE; k++)
        if (output_spikes[k]) counts[k] <= counts[k] + SPIKE_COUNT_WIDTH'(1);
    end
  end

endmodule
