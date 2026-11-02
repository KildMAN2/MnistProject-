// -----------------------------------------------------------------------------
// decay_19_20 : exact membrane decay  decayed = round_ties_to_even(19*m / 20)
//
// RTL milestone 1 - CORRECTNESS / REFERENCE VERSION.
// Purely combinational, no pipeline registers, no '*' or '/' operators, no DSPs.
// Algorithm is a direct port of hw_decay_shiftadd() in
// fpga/scripts/lif_equiv_check.py, which is bit-exact against
// _round_ratio_ties_to_even(m*19, 20) in software/quantization/quantization.py.
//
// Width plan (W = MEM_WIDTH):
//   a      : W      unsigned  |m|, up to 2^(W-1)  (W bits are needed for -2^(W-1))
//   x=19a  : W+5    unsigned  19*2^(W-1) < 2^(W+4)
//   y=x>>2 : W+3    unsigned  floor(x/4)  (floor(y/5) == floor(x/20))
//   q*     : W+3    unsigned  approximations of y/5, never above the true value
//   r4     : W+3    unsigned  y - 5q  (observed max 10 before fix-up)
//   rem    : 3 bits          y - 5q after fix-up, 0..4
//   r      : 5 bits          x mod 20 = {rem, x[1:0]}, 0..19
//   mag    : W+3    unsigned  rounded |19m/20| <= 0.95*2^(W-1)+1 < 2^(W-1)
//   out    : W      signed
// Valid for 2 <= MEM_WIDTH <= 60 (the shift ladder 4/8/16/32 covers y < 2^64).
// -----------------------------------------------------------------------------
module decay_19_20 #(
    parameter int MEM_WIDTH = 38
) (
    input  logic signed [MEM_WIDTH-1:0] membrane_in,
    output logic signed [MEM_WIDTH-1:0] decayed_out
);

  localparam int W  = MEM_WIDTH;
  localparam int XW = W + 5;
  localparam int YW = W + 3;

  initial begin
    if (W < 2 || W > 60) $fatal(1, "decay_19_20: MEM_WIDTH=%0d outside supported range 2..60", W);
  end

  // ---- sign and magnitude (safe for the most-negative value) ---------------
  logic         sign;
  logic [W-1:0] a;              // |m| as W-bit unsigned: -2^(W-1) -> 2^(W-1) fits
  assign sign = membrane_in[W-1];
  assign a    = sign ? (~membrane_in + {{(W-1){1'b0}}, 1'b1}) : membrane_in;

  // ---- x = 19*a = (a<<4) + (a<<1) + a --------------------------------------
  logic [XW-1:0] a_x, x;
  assign a_x = {{5{1'b0}}, a};
  assign x   = (a_x << 4) + (a_x << 1) + a_x;

  // ---- y = x >> 2 -----------------------------------------------------------
  logic [YW-1:0] y;
  assign y = x[XW-1:2];

  // ---- divide-by-5 approximation by shifts/adds (never exceeds floor(y/5)) --
  logic [YW-1:0] q1, q2, q3, q4, q5, q6;
  assign q1 = (y >> 1) + (y >> 2);
  assign q2 = q1 + (q1 >> 4);
  assign q3 = q2 + (q2 >> 8);
  assign q4 = q3 + (q3 >> 16);
  assign q5 = q4 + (q4 >> 32);
  assign q6 = q5 >> 2;

  // ---- exact fix-up: r4 = y - 5*q6, add up to 4 more fives ------------------
  logic [YW-1:0] q6_x5, r4;
  assign q6_x5 = (q6 << 2) + q6;
  assign r4    = y - q6_x5;

  logic [2:0] fix;
  always_comb begin
    fix = 3'd0;
    if (r4 >= 5)  fix = fix + 3'd1;
    if (r4 >= 10) fix = fix + 3'd1;
    if (r4 >= 15) fix = fix + 3'd1;
    if (r4 >= 20) fix = fix + 3'd1;
  end

  logic [YW-1:0] q;             // q == floor(x/20)
  assign q = q6 + {{(YW-3){1'b0}}, fix};

  // ---- exact remainder x mod 20 --------------------------------------------
  logic [YW-1:0] q_x5, rem_y;
  assign q_x5  = (q << 2) + q;
  assign rem_y = y - q_x5;      // 0..4

  logic [4:0] r;                // x mod 20 = 4*rem_y + x[1:0]
  assign r = {rem_y[2:0], x[1:0]};

  // ---- round to nearest, ties to even --------------------------------------
  logic inc;
  assign inc = (r > 5'd10) | ((r == 5'd10) & q[0]);

  logic [YW-1:0] mag;
  assign mag = q + {{(YW-1){1'b0}}, inc};

  // ---- restore sign ---------------------------------------------------------
  logic [W-1:0] mag_w;
  assign mag_w = mag[W-1:0];    // mag < 2^(W-1), upper bits are zero
  assign decayed_out = sign ? (~mag_w + {{(W-1){1'b0}}, 1'b1}) : mag_w;

endmodule
