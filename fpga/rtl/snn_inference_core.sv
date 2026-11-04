// -----------------------------------------------------------------------------
// snn_inference_core : functional integration of all verified blocks
// (RTL milestone 12). Simulation wrapper, NOT the board-specific snn_top.
//
//   pixel write port -> l1_current_sweep (pixel buffer, cur1 cache, computed ONCE)
//   cur1 cache       -> l1_lif_sweep (25 reads of the same cache)
//   hidden_spikes    -> l2_current_engine -> cur2 -> l2_lif_sweep -> output_spikes
//   output_spikes    -> output_spike_counters -> counts -> argmax10
//   snn_inference_controller sequences everything with done handshakes.
// -----------------------------------------------------------------------------
module snn_inference_core
  import snn_params_pkg::*;
(
    input  logic                          clk,
    input  logic                          rst,

    input  logic                          pixel_we,
    input  logic [9:0]                    pixel_waddr,
    input  logic [7:0]                    pixel_wdata,

    input  logic                          start,
    output logic                          busy,
    output logic                          done,
    output logic [3:0]                    predicted_digit,

    output logic [SPIKE_COUNT_WIDTH-1:0]  counts [0:OUTPUT_SIZE-1]
);

  // controller <-> children
  logic l1_current_start, l1_current_done, l1_current_busy;
  logic l1_clear_state, l1_clear_done, l1_start_step, l1_step_done, l1_busy;
  logic l2_clear_state, l2_clear_done, l2_current_start, l2_current_done, l2_current_busy;
  logic l2_start_step, l2_step_done, l2_busy;
  logic clear_counts, count_enable;
  logic [4:0] timestep;

  logic [7:0]                     cur1_raddr;
  logic [L1_ACC_WIDTH-1:0]        cur1_rdata;
  logic [HIDDEN_SIZE-1:0]         hidden_spikes;
  logic signed [L2_ACC_WIDTH-1:0] cur2 [0:OUTPUT_SIZE-1];
  logic [OUTPUT_SIZE-1:0]         output_spikes;
  logic [3:0]                     argmax_prediction;
  logic [SPIKE_COUNT_WIDTH-1:0]   max_count;

  l1_current_sweep u_l1c (
      .clk(clk), .rst(rst),
      .pixel_we(pixel_we), .pixel_waddr(pixel_waddr), .pixel_wdata(pixel_wdata),
      .start(l1_current_start), .busy(l1_current_busy), .done(l1_current_done),
      .cur1_raddr(cur1_raddr), .cur1_rdata(cur1_rdata));

  l1_lif_sweep u_l1s (
      .clk(clk), .rst(rst),
      .clear_state(l1_clear_state), .clear_done(l1_clear_done),
      .start_step(l1_start_step), .busy(l1_busy), .step_done(l1_step_done),
      .cur1_raddr(cur1_raddr), .cur1_rdata(cur1_rdata),
      .hidden_spikes(hidden_spikes));

  l2_current_engine u_l2c (
      .clk(clk), .rst(rst),
      .start(l2_current_start), .hidden_spikes(hidden_spikes),
      .busy(l2_current_busy), .done(l2_current_done), .cur2(cur2));

  l2_lif_sweep u_l2s (
      .clk(clk), .rst(rst),
      .clear_state(l2_clear_state), .clear_done(l2_clear_done),
      .start_step(l2_start_step), .busy(l2_busy), .step_done(l2_step_done),
      .cur2(cur2), .output_spikes(output_spikes));

  output_spike_counters u_cnt (
      .clk(clk), .rst(rst),
      .clear_counts(clear_counts), .count_enable(count_enable),
      .output_spikes(output_spikes), .counts(counts));

  argmax10 u_argmax (
      .counts(counts), .predicted_digit(argmax_prediction), .max_count(max_count));

  snn_inference_controller u_ctrl (
      .clk(clk), .rst(rst), .start(start),
      .l1_current_done(l1_current_done), .l1_clear_done(l1_clear_done), .l1_step_done(l1_step_done),
      .l2_clear_done(l2_clear_done), .l2_current_done(l2_current_done), .l2_step_done(l2_step_done),
      .argmax_prediction(argmax_prediction),
      .busy(busy), .done(done), .predicted_digit(predicted_digit), .timestep(timestep),
      .l1_current_start(l1_current_start), .l1_clear_state(l1_clear_state), .l1_start_step(l1_start_step),
      .l2_clear_state(l2_clear_state), .l2_current_start(l2_current_start), .l2_start_step(l2_start_step),
      .clear_counts(clear_counts), .count_enable(count_enable));

endmodule
