// -----------------------------------------------------------------------------
// l1_mac : Layer-1 multiply-accumulate datapath for ONE hidden neuron
// (RTL milestone 3).
//
// Golden equation (Int16SNN):  current[j] = bias1[j] + sum_{i=0..783} pixel[i]*weight1[j][i]
//
// Convention (bias-first):
//   load_bias : acc <= sign_extend(bias)                       (starts a neuron)
//   valid     : acc <= acc + sign_extend(pixel_s * weight)     (one pair per clock)
// After the clock edge that accumulates the last (784th) product, acc_out
// already equals bias + sum(products). The block does NOT count inputs; the
// future controller owns the 0..783 input index.
//
// Numeric types (all explicit):
//   pixel    logic        [7:0]   0..255 (raw MNIST byte)
//   pixel_s  logic signed [8:0]   {1'b0, pixel}        0..255
//   weight   logic signed [15:0]  -32768..32767
//   product  logic signed [23:0]  -8,355,840 .. 8,355,585  (fits: |min| < 2^23)
//   bias     logic signed [31:0]  int32
//   acc      logic signed [ACC_WIDTH-1:0], ACC_WIDTH = 35 (generic-safe bound)
// There is no saturation: exact signed arithmetic inside the proven width.
//
// Timing semantics: acc_out is a register. The accumulate/load happens at the
// clock edge (nonblocking), so the new value is visible only AFTER that edge.
// result_valid is a registered copy of "an accumulate happened at the previous
// edge" -> while result_valid is 1, acc_out includes the product presented
// during the previous clock cycle.
//
// Priority: rst > load_bias > valid. load_bias together with valid loads the
// bias and discards that cycle's product. With neither, acc_out holds.
//
// This is the correctness/reference version: multiply and add in one clock
// stage (no pipeline registers). The future DSP-pipelined version must keep the
// same arithmetic.
// -----------------------------------------------------------------------------
module l1_mac #(
    parameter int ACC_WIDTH = 35
) (
    input  logic                        clk,
    input  logic                        rst,          // synchronous, active high
    input  logic                        load_bias,    // start a neuron: acc <= bias
    input  logic signed [31:0]          bias,
    input  logic                        valid,        // accumulate this pixel*weight
    input  logic        [7:0]           pixel,
    input  logic signed [15:0]          weight,
    output logic signed [ACC_WIDTH-1:0] acc_out,
    output logic                        result_valid
);

  localparam int PROD_WIDTH = 24;

  initial begin
    if (ACC_WIDTH < 32)
      $fatal(1, "l1_mac: ACC_WIDTH (%0d) must hold the 32-bit bias", ACC_WIDTH);
  end

  // ---- explicit pixel conversion: unsigned 0..255 -> signed 9 bit -------------
  logic signed [8:0] pixel_s;
  assign pixel_s = {1'b0, pixel};

  // ---- signed multiply: both operands signed, so the 9-bit pixel and the
  //      16-bit weight are sign-extended to the 24-bit result context ----------
  logic signed [PROD_WIDTH-1:0] product;
  assign product = pixel_s * weight;

  // ---- explicit sign extension to the accumulator width ------------------------
  logic signed [ACC_WIDTH-1:0] product_ext, bias_ext;
  assign product_ext = {{(ACC_WIDTH - PROD_WIDTH) {product[PROD_WIDTH-1]}}, product};
  assign bias_ext    = {{(ACC_WIDTH - 32) {bias[31]}}, bias};

  // ---- accumulator register -----------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      acc_out      <= '0;
      result_valid <= 1'b0;
    end else begin
      result_valid <= valid & ~load_bias;
      if (load_bias) acc_out <= bias_ext;
      else if (valid) acc_out <= acc_out + product_ext;
    end
  end

endmodule
