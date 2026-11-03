// Primitive self-checking testbench for l2_acc (RTL milestone 8).
// Run from fpga/tb:
//   vlog -sv ../rtl/l2_acc.sv tb_l2_acc.sv
//   vsim -c work.tb_l2_acc -do "run -all; quit -f"
// Stimulus is applied 1 ns after posedge; results are checked after the next edge
// (#1), when the nonblocking update is visible.
`timescale 1ns/1ps

module tb_l2_acc;
  localparam int AW = 33;

  logic clk = 1'b0;
  always #5 clk = ~clk;

  logic rst, load_bias, valid, spike;
  logic signed [31:0] bias;
  logic signed [15:0] weight;
  logic signed [AW-1:0] acc_out;
  logic result_valid;

  l2_acc #(.ACC_WIDTH(AW)) dut (.*);

  longint ref_acc;
  int tests, fails, valid_checks, valid_fail;

  task automatic step(input bit lb, input bit v, input bit s, input int w, input int b);
    load_bias = lb; valid = v; spike = s; weight = 16'(w); bias = 32'(b);
    @(posedge clk); #1;
    load_bias = 1'b0; valid = 1'b0;
  endtask

  task automatic chk(input string what);
    tests++;
    if (acc_out !== AW'(ref_acc)) begin
      fails++;
      $display("FAIL %s: acc=%0d exp=%0d", what, acc_out, ref_acc);
    end
  endtask

  task automatic do_bias(input int b);
    step(1, 0, 0, 0, b);
    ref_acc = longint'(b);
    chk("load_bias");
  endtask

  task automatic do_acc(input bit s, input int w);
    bit v_expected;
    step(0, 1, s, w, 0);
    if (s) ref_acc += longint'(w);
    chk(s ? "add" : "hold");
    valid_checks++;
    if (result_valid !== 1'b1) begin valid_fail++; fails++; $display("result_valid low after valid (spike=%0b)", s); end
  endtask

  initial begin
    int w;
    rst = 1'b1; load_bias = 0; valid = 0; spike = 0; bias = 0; weight = 0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    ref_acc = 0; chk("reset");
    if (result_valid !== 1'b0) begin fails++; $display("result_valid not reset"); end

    // spike=0 leaves the accumulator unchanged (positive and negative weight)
    do_bias(1000);
    do_acc(0, 5);   do_acc(0, -5);   do_acc(0, 32767);   do_acc(0, -32768);
    // spike=1 adds the signed weight
    do_acc(1, 5);   do_acc(1, -5);   do_acc(1, 7);       do_acc(1, -1000);
    // extreme weights
    do_acc(1, 32767);   do_acc(1, -32768);   do_acc(1, -32768);   do_acc(1, 32767);
    // positive / negative bias and int32 extremes
    do_bias(123456);   do_acc(1, -32768);
    do_bias(-123456);  do_acc(1, 32767);
    do_bias(32'sh7FFFFFFF);  do_acc(1, 32767);  do_acc(0, 32767);
    do_bias(32'sh80000000);  do_acc(1, -32768); do_acc(1, -1);
    // worst-case sums that reach the bound: bias max + 256 * 32767, bias min + 256 * -32768
    do_bias(32'sh7FFFFFFF);
    for (int i = 0; i < 256; i++) do_acc(1, 32767);
    if (ref_acc != 64'sd2155871999) begin fails++; $display("bound arithmetic wrong"); end
    do_bias(32'sh80000000);
    for (int i = 0; i < 256; i++) do_acc(1, -32768);
    if (ref_acc != -64'sd2155872256) begin fails++; $display("bound arithmetic wrong"); end
    // multiple sequential additions, mixed spikes
    do_bias(-7);
    for (int i = 0; i < 500; i++) begin
      w = $urandom_range(0, 65535) - 32768;
      do_acc($urandom_range(0, 1), w);
    end
    // random biases
    for (int n = 0; n < 200; n++) begin
      do_bias($urandom);
      for (int i = 0; i < 20; i++) begin
        w = $urandom_range(0, 65535) - 32768;
        do_acc($urandom_range(0, 1), w);
      end
    end

    // result_valid: low when idle or while loading the bias, high after a consume
    step(0, 0, 0, 0, 0);
    valid_checks++; if (result_valid !== 1'b0) begin valid_fail++; fails++; $display("result_valid high while idle"); end
    step(1, 1, 1, 100, 5);                     // load_bias wins over valid: bias loaded, product discarded
    ref_acc = 5; chk("load_bias priority");
    valid_checks++; if (result_valid !== 1'b0) begin valid_fail++; fails++; $display("result_valid high on load_bias"); end

    // reset clears accumulator
    rst = 1'b1; @(posedge clk); #1 rst = 1'b0;
    ref_acc = 0; chk("reset clears acc");

    $display("primitive l2_acc tests: %0d (result_valid checks %0d)", tests, valid_checks);
    $display("total failures: %0d", fails);
    if (fails == 0) $display("PASS tb_l2_acc");
    else $fatal(1, "FAIL tb_l2_acc");
    $finish;
  end
endmodule
