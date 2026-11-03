// -----------------------------------------------------------------------------
// l2_acc : Layer-2 accumulate datapath for ONE output neuron (RTL milestone 8).
// Multiplier-free equivalent of l1_mac: the hidden activations are binary spikes.
//
// Golden equation (Int16SNN):
//   cur2[k] = bias2[k] + sum_{j=0..255} hidden_spike[j] * weight2[k][j]
//
//   load_bias : acc <= sign_extend(bias)                    (starts an output neuron)
//   valid     : "consume one hidden index"
//                 spike = 1 : acc <= acc + sign_extend(weight)
//                 spike = 0 : acc <= acc          (the index is still consumed)
// The block does not count inputs; the engine owns the 0..255 index.
//
// Numeric types (all explicit)
//   weight  logic signed [15:0]            -32768..32767
//   bias    logic signed [31:0]            int32
//   acc     logic signed [ACC_WIDTH-1:0]   ACC_WIDTH = 33
// Width proof (N_INPUTS = 256 terms):
//   max =  2^31-1 + 256*32767 =  2,155,871,999 < 2^32 = 4,294,967,296
//   min = -2^31   - 256*32768 = -2,155,872,256 >= -2^32 = -4,294,967,296
//   (the bound already exceeds 2^31, so 32 bits are not enough)
//   => signed 33 bits ([-2^32, 2^32-1]) is sufficient. Checked at elaboration.
// There is no saturation.
//
// Timing: acc_out is a register, new value visible after the clock edge.
// result_valid is a registered copy of "an index was consumed at the previous
// edge" (valid & ~load_bias), set for spike = 0 as well as spike = 1.
// Priority: rst > load_bias > valid.
// -----------------------------------------------------------------------------
module l2_acc #(
    parameter int ACC_WIDTH = 33,
    parameter int N_INPUTS  = 256
) (
    input  logic                        clk,
    input  logic                        rst,        // synchronous, active high
    input  logic                        load_bias,
    input  logic signed [31:0]          bias,
    input  logic                        valid,
    input  logic                        spike,
    input  logic signed [15:0]          weight,
    output logic signed [ACC_WIDTH-1:0] acc_out,
    output logic                        result_valid
);

  initial begin
    longint maxv, minv, lim;
    maxv = 64'sd2147483647 + N_INPUTS * 64'sd32767;
    minv = -64'sd2147483648 - N_INPUTS * 64'sd32768;
    lim  = 64'sd1 <<< (ACC_WIDTH - 1);
    if (ACC_WIDTH < 33 || ACC_WIDTH > 62 || maxv >= lim || minv < -lim)
      $fatal(1, "l2_acc: ACC_WIDTH %0d cannot hold bias + %0d int16 terms", ACC_WIDTH, N_INPUTS);
  end

  // ---- explicit sign extension ----------------------------------------------------
  logic signed [ACC_WIDTH-1:0] weight_ext, bias_ext;
  assign weight_ext = {{(ACC_WIDTH - 16) {weight[15]}}, weight};
  assign bias_ext   = {{(ACC_WIDTH - 32) {bias[31]}}, bias};

  always_ff @(posedge clk) begin
    if (rst) begin
      acc_out      <= '0;
      result_valid <= 1'b0;
    end else begin
      result_valid <= valid & ~load_bias;
      if (load_bias) acc_out <= bias_ext;
      else if (valid && spike) acc_out <= acc_out + weight_ext;
    end
  end

endmodule
