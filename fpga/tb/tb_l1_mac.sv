// Self-checking testbench for l1_mac (RTL milestone 3).
// Vectors: fpga/scripts/generate_mac_vectors.py. Run with working directory = fpga/tb.
//
// Clock semantics: stimulus is applied 1 ns after a rising edge. The DUT
// registers update with nonblocking assignments at the edge, so the new
// accumulator value is only visible AFTER the edge's NBA region. The checks
// therefore run #1 after @(posedge clk). CHECK_EARLY=1 deliberately samples in
// the same time step as the edge (before the NBA update) to prove that an
// early check is detected.
`timescale 1ns / 1ps

module tb_l1_mac #(
    parameter bit CHECK_EARLY = 0
);

  localparam int ACC_W = 35;

  logic                     clk = 0;
  logic                     rst;
  logic                     load_bias;
  logic signed [31:0]       bias;
  logic                     valid;
  logic        [7:0]        pixel;
  logic signed [15:0]       weight;
  logic signed [ACC_W-1:0]  acc_out;
  logic                     result_valid;

  l1_mac #(.ACC_WIDTH(ACC_W)) dut (.*);

  always #5 clk = ~clk;

  integer failures = 0;
  integer prim_tests = 0, neuron_tests = 0, directed_tests = 0;
  integer mac_cycles = 0;      // cycles with an accumulate (valid=1)
  integer clock_cycles = 0;    // all clock edges after reset release
  integer overflow_flags = 0;

  // wide reference accumulator kept in the testbench (64-bit)
  longint ref_acc;

  function automatic bit fits35(input longint v);
    return (v >= -(64'sd1 <<< (ACC_W - 1))) && (v <= ((64'sd1 <<< (ACC_W - 1)) - 1));
  endfunction

  // one clock: apply inputs (already driven), wait for the edge, then sample
  task automatic tick;
    begin
      @(posedge clk);
      clock_cycles = clock_cycles + 1;
      if (!CHECK_EARLY) #1;
    end
  endtask

  task automatic after_check;
    begin
      if (CHECK_EARLY) #1;
    end
  endtask

  task automatic idle_inputs;
    begin
      load_bias = 0; valid = 0; pixel = 0; weight = 0; bias = 0;
    end
  endtask

  task automatic expect_acc(input string what, input longint expected);
    begin
      if (acc_out !== expected[ACC_W-1:0]) begin
        failures = failures + 1;
        if (failures <= 20)
          $display("MISMATCH %s: expected=%0d (0x%h) actual=%0d (0x%h) pixel=%0d weight=%0d",
                   what, expected, expected[ACC_W-1:0], acc_out, acc_out, pixel, weight);
      end
    end
  endtask

  // load bias for one clock
  task automatic do_load_bias(input logic signed [31:0] b);
    begin
      idle_inputs();
      load_bias = 1; bias = b;
      tick();
      ref_acc = b;
      expect_acc("load_bias", ref_acc);
      after_check();
      if (result_valid !== 1'b0 && !CHECK_EARLY) begin
        failures = failures + 1;
        $display("MISMATCH: result_valid must be 0 after a bias load");
      end
      idle_inputs();
    end
  endtask

  // one accumulate clock; checks the register after the edge
  task automatic do_mac(input logic [7:0] p, input logic signed [15:0] w);
    begin
      idle_inputs();
      valid = 1; pixel = p; weight = w;
      tick();
      mac_cycles = mac_cycles + 1;
      ref_acc = ref_acc + longint'($signed({1'b0, p})) * longint'(w);
      if (!fits35(ref_acc)) begin
        overflow_flags = overflow_flags + 1;
        $display("OVERFLOW: reference accumulator %0d leaves signed 35 bits", ref_acc);
      end
      expect_acc("accumulate", ref_acc);
      if (!CHECK_EARLY && result_valid !== 1'b1) begin
        failures = failures + 1;
        $display("MISMATCH: result_valid must be 1 after an accumulate");
      end
      after_check();
      idle_inputs();
    end
  endtask

  integer fd, rc, i, t;
  logic [7:0]            v_pix;
  logic [15:0]           v_w;
  logic [23:0]           v_prod;
  logic [31:0]           v_bias;
  logic [ACC_W-1:0]      v_exp;
  logic signed [ACC_W-1:0] cur_before;

  initial begin
    rst = 1; idle_inputs();
    #1;
    repeat (3) @(posedge clk);
    #1;
    // ---- 1. reset ---------------------------------------------------------------
    if (acc_out !== '0 || result_valid !== 1'b0) begin
      failures = failures + 1;
      $display("MISMATCH: reset did not clear acc/result_valid (acc=%0d rv=%0b)", acc_out, result_valid);
    end
    directed_tests = directed_tests + 1;
    rst = 0;

    // ---- 2. bias initialisation (positive, negative, extremes) ------------------
    do_load_bias(32'sd1000);        directed_tests = directed_tests + 1;
    do_load_bias(-32'sd1000);       directed_tests = directed_tests + 1;
    do_load_bias(32'sh7FFFFFFF);    directed_tests = directed_tests + 1;
    do_load_bias(32'sh80000000);    directed_tests = directed_tests + 1;

    // ---- 3-9. single multiplications and accumulations ----------------------------
    do_load_bias(0);
    do_mac(8'd5, 16'sd7);                       // 35, one multiplication
    do_mac(8'd10, 16'sd3);                      // 65, multiple accumulations
    do_mac(8'd200, -16'sd1);                    // negative weight: -200 -> -135
    do_mac(8'd0, -16'sh8000);                   // zero pixel: unchanged
    do_mac(8'd255, 16'sh7FFF);                  // largest pixel, max weight
    do_mac(8'd255, -16'sh8000);                 // largest pixel, min weight
    do_mac(8'd1, -16'sd1);
    directed_tests = directed_tests + 7;

    // valid=0 holds the accumulator and clears result_valid
    cur_before = acc_out;
    idle_inputs(); pixel = 8'd99; weight = 16'sd99;   // data present but valid=0
    tick();
    if (!CHECK_EARLY) begin
      if (acc_out !== cur_before) begin failures = failures + 1; $display("MISMATCH: acc changed with valid=0"); end
      if (result_valid !== 1'b0)  begin failures = failures + 1; $display("MISMATCH: result_valid stuck"); end
    end
    after_check();
    directed_tests = directed_tests + 1;

    // load_bias has priority over valid in the same cycle
    idle_inputs(); load_bias = 1; bias = 32'sd777; valid = 1; pixel = 8'd50; weight = 16'sd50;
    tick();
    ref_acc = 777;
    expect_acc("load_bias priority over valid", ref_acc);
    after_check();
    idle_inputs();
    directed_tests = directed_tests + 1;

    // synchronous reset in the middle of work
    rst = 1; tick(); rst = 0;
    if (!CHECK_EARLY && (acc_out !== '0 || result_valid !== 1'b0)) begin
      failures = failures + 1; $display("MISMATCH: mid-run reset");
    end
    after_check();
    directed_tests = directed_tests + 1;

    // ---- primitive multiplication vectors from Python -----------------------------
    fd = $fopen("vectors/mac_prim_vectors.txt", "r");
    if (fd == 0) $fatal(1, "cannot open vectors/mac_prim_vectors.txt (run from fpga/tb)");
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%h %h %h\n", v_pix, v_w, v_prod);
      if (rc == 3) begin
        do_load_bias(0);
        do_mac(v_pix, v_w);
        // independent check against the Python product (sign-extended 24 -> 64)
        if (!CHECK_EARLY && acc_out !== {{(ACC_W-24){v_prod[23]}}, v_prod}) begin
          failures = failures + 1;
          if (failures <= 20)
            $display("MISMATCH primitive #%0d: pixel=%0d weight=%0d expected=%0d actual=%0d",
                     prim_tests, v_pix, $signed(v_w), $signed(v_prod), acc_out);
        end
        prim_tests = prim_tests + 1;
      end
    end
    $fclose(fd);

    // ---- complete neurons ------------------------------------------------------------
    fd = $fopen("vectors/mac_neuron_vectors.txt", "r");
    if (fd == 0) $fatal(1, "cannot open vectors/mac_neuron_vectors.txt (run from fpga/tb)");
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%h %h\n", v_bias, v_exp);
      if (rc == 2) begin
        do_load_bias($signed(v_bias));
        for (i = 0; i < 784; i = i + 1) begin
          rc = $fscanf(fd, "%h %h\n", v_pix, v_w);
          do_mac(v_pix, v_w);
        end
        // final result vs the Python expected current (after the last edge)
        if (!CHECK_EARLY && acc_out !== v_exp) begin
          failures = failures + 1;
          $display("MISMATCH neuron test #%0d: expected=%0d (0x%h) actual=%0d (0x%h) bias=%0d",
                   neuron_tests, $signed(v_exp), v_exp, acc_out, acc_out, $signed(v_bias));
        end
        if (CHECK_EARLY && acc_out === v_exp) begin
          // early sampling saw the final value: cannot happen unless the DUT is combinational
        end
        neuron_tests = neuron_tests + 1;
      end
    end
    $fclose(fd);

    $display("directed tests    : %0d", directed_tests);
    $display("primitive tests   : %0d", prim_tests);
    $display("neuron tests      : %0d (x784 products)", neuron_tests);
    $display("MAC (valid) cycles: %0d", mac_cycles);
    $display("total clock cycles: %0d", clock_cycles);
    $display("overflow flags    : %0d", overflow_flags);
    $display("failures          : %0d", failures);
    if (prim_tests == 0 || neuron_tests == 0) $fatal(1, "no vectors read");
    if (failures == 0 && overflow_flags == 0) $display("PASS: tb_l1_mac");
    else $fatal(1, "FAIL: tb_l1_mac (%0d failures, %0d overflows)", failures, overflow_flags);
    $finish;
  end

  // watchdog
  initial begin
    #2000000000;
    $fatal(1, "timeout");
  end

endmodule
