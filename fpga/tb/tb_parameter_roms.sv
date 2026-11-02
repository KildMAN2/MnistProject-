// Self-checking testbench for the parameter ROMs (RTL milestone 4).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv tb_parameter_roms.sv
//   vsim -c work.tb_parameter_roms -do "run -all; quit -f"
// EARLY=1 samples the data BEFORE the capturing clock edge (must FAIL: proves the
// 1-cycle latency check has teeth).
`timescale 1ns/1ps

module tb_parameter_roms #(
    parameter bit EARLY = 1'b0
);
  import snn_params_pkg::*;

  localparam int MAXV = 20000;

  logic clk = 1'b0;
  always #5 clk = ~clk;

  logic [L1_WEIGHT_ADDR_WIDTH-1:0] a_l1w;
  logic [L1_BIAS_ADDR_WIDTH-1:0]   a_l1b;
  logic [L2_WEIGHT_ADDR_WIDTH-1:0] a_l2w;
  logic [L2_BIAS_ADDR_WIDTH-1:0]   a_l2b;
  logic [WEIGHT_WIDTH-1:0] d_l1w, d_l2w;
  logic [BIAS_WIDTH-1:0]   d_l1b, d_l2b;

  sync_rom #(.WIDTH(WEIGHT_WIDTH), .DEPTH(L1_WEIGHT_DEPTH), .ADDR_WIDTH(L1_WEIGHT_ADDR_WIDTH),
             .INIT_FILE(L1_WEIGHT_FILE)) u_l1w (.clk(clk), .addr(a_l1w), .data(d_l1w));
  sync_rom #(.WIDTH(BIAS_WIDTH), .DEPTH(L1_BIAS_DEPTH), .ADDR_WIDTH(L1_BIAS_ADDR_WIDTH),
             .INIT_FILE(L1_BIAS_FILE)) u_l1b (.clk(clk), .addr(a_l1b), .data(d_l1b));
  sync_rom #(.WIDTH(WEIGHT_WIDTH), .DEPTH(L2_WEIGHT_DEPTH), .ADDR_WIDTH(L2_WEIGHT_ADDR_WIDTH),
             .INIT_FILE(L2_WEIGHT_FILE)) u_l2w (.clk(clk), .addr(a_l2w), .data(d_l2w));
  sync_rom #(.WIDTH(BIAS_WIDTH), .DEPTH(L2_BIAS_DEPTH), .ADDR_WIDTH(L2_BIAS_ADDR_WIDTH),
             .INIT_FILE(L2_BIAS_FILE)) u_l2b (.clk(clk), .addr(a_l2b), .data(d_l2b));

  // small ROMs holding the signedness corner cases
  logic [2:0] a_s16;
  logic [2:0] a_s32;
  logic [15:0] d_s16;
  logic [31:0] d_s32;
  sync_rom #(.WIDTH(16), .DEPTH(8), .ADDR_WIDTH(3),
             .INIT_FILE("vectors/rom_signed_test16.mem")) u_s16 (.clk(clk), .addr(a_s16), .data(d_s16));
  sync_rom #(.WIDTH(32), .DEPTH(8), .ADDR_WIDTH(3),
             .INIT_FILE("vectors/rom_signed_test32.mem")) u_s32 (.clk(clk), .addr(a_s32), .data(d_s32));

  int           vt [0:MAXV-1];
  int           va [0:MAXV-1];
  logic [31:0]  vh [0:MAXV-1];
  int           nvec;
  logic [15:0]  l2w_exp [0:L2_WEIGHT_DEPTH-1];

  int tests   [0:3];
  int fails   [0:3];
  int neg_checked;
  int signed_tests, signed_fails;
  int latency_tests, latency_fails;
  int stream_tests, stream_fails;
  string names [0:3] = '{"L1 weight", "L1 bias", "L2 weight", "L2 bias"};

  function automatic longint sx16(input logic [15:0] v);
    return longint'($signed(v));
  endfunction
  function automatic longint sx32(input logic [31:0] v);
    return longint'($signed(v));
  endfunction

  task automatic check_rom(input int t, input int addr, input logic [31:0] exp);
    logic [31:0] act;
    logic [31:0] e;
    int w;
    longint act_s, exp_s;
    case (t)
      0: begin act = {16'b0, d_l1w}; w = 16; end
      1: begin act = d_l1b;          w = 32; end
      2: begin act = {16'b0, d_l2w}; w = 16; end
      default: begin act = d_l2b;    w = 32; end
    endcase
    e = (w == 16) ? {16'b0, exp[15:0]} : exp;
    act_s = (w == 16) ? sx16(act[15:0]) : sx32(act);
    exp_s = (w == 16) ? sx16(e[15:0])   : sx32(e);
    tests[t]++;
    if (act !== e || act_s != exp_s) begin
      fails[t]++;
      if (fails[t] <= 10)
        $display("MISMATCH %s addr=%0d expected raw=%h signed=%0d actual raw=%h signed=%0d",
                 names[t], addr, e, exp_s, act, act_s);
    end
    if (exp_s < 0) begin
      neg_checked++;
      if (act_s >= 0) begin
        fails[t]++;
        $display("SIGN ERROR %s addr=%0d negative parameter read as non-negative", names[t], addr);
      end
    end
  endtask

  task automatic drive(input int t, input int addr);
    case (t)
      0: a_l1w = addr[L1_WEIGHT_ADDR_WIDTH-1:0];
      1: a_l1b = addr[L1_BIAS_ADDR_WIDTH-1:0];
      2: a_l2w = addr[L2_WEIGHT_ADDR_WIDTH-1:0];
      default: a_l2b = addr[L2_BIAS_ADDR_WIDTH-1:0];
    endcase
  endtask

  task automatic read_and_check(input int t, input int addr, input logic [31:0] exp);
    drive(t, addr);
    if (EARLY) #1;          // sample before the capturing edge
    else begin
      @(posedge clk);
      #1;                    // let the nonblocking update settle
    end
    check_rom(t, addr, exp);
  endtask

  task automatic expect_s(input bit wide, input int idx, input logic [31:0] exp_raw,
                          input longint exp_signed);
    logic [31:0] act;
    longint act_s;
    if (wide) begin a_s32 = idx[2:0]; end else begin a_s16 = idx[2:0]; end
    @(posedge clk); #1;
    act = wide ? d_s32 : {16'b0, d_s16};
    act_s = wide ? sx32(d_s32) : sx16(d_s16);
    signed_tests++;
    if (act !== exp_raw || act_s != exp_signed) begin
      signed_fails++;
      $display("SIGNED MISMATCH wide=%0d idx=%0d raw=%h (exp %h) signed=%0d (exp %0d)",
               wide, idx, act, exp_raw, act_s, exp_signed);
    end
  endtask

  int fd, rc, k, total_fail;
  int tt, aa;
  logic [31:0] hh;
  logic [15:0] before_val;

  initial begin
    for (k = 0; k < 4; k++) begin tests[k] = 0; fails[k] = 0; end
    neg_checked = 0; signed_tests = 0; signed_fails = 0;
    latency_tests = 0; latency_fails = 0; stream_tests = 0; stream_fails = 0;
    a_l1w = '0; a_l1b = '0; a_l2w = '0; a_l2b = '0; a_s16 = '0; a_s32 = '0;

    // package consistency
    if (L1_WEIGHT_DEPTH != INPUT_SIZE*HIDDEN_SIZE || L2_WEIGHT_DEPTH != HIDDEN_SIZE*OUTPUT_SIZE ||
        (1 << L1_WEIGHT_ADDR_WIDTH) < L1_WEIGHT_DEPTH || (1 << (L1_WEIGHT_ADDR_WIDTH-1)) >= L1_WEIGHT_DEPTH ||
        (1 << L2_WEIGHT_ADDR_WIDTH) < L2_WEIGHT_DEPTH || (1 << (L2_WEIGHT_ADDR_WIDTH-1)) >= L2_WEIGHT_DEPTH)
      $fatal(1, "snn_params_pkg depth/address-width inconsistency");

    // load golden vectors
    fd = $fopen("vectors/rom_reference_vectors.txt", "r");
    if (fd == 0) $fatal(1, "cannot open vectors/rom_reference_vectors.txt");
    nvec = 0;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%d %d %h\n", tt, aa, hh);
      if (rc == 3) begin
        vt[nvec] = tt; va[nvec] = aa; vh[nvec] = hh;
        if (tt == 2) l2w_exp[aa] = hh[15:0];
        nvec++;
      end
    end
    $fclose(fd);
    $display("loaded %0d golden ROM vectors", nvec);

    repeat (3) @(posedge clk);
    #1;

    // ---- directed boundary reads (independent of file ordering) ----
    begin : directed
      int l1_addrs [0:10];
      int l2_addrs [0:5];
      int idx, found;
      l1_addrs = '{0, 1, 783, 784, 785, 127*784, 128*784, 255*784, 200703, 255*784+783, 1567};
      l2_addrs = '{0, 255, 256, 257, 9*256, 2559};
      for (k = 0; k < 11; k++) begin
        found = 0;
        for (idx = 0; idx < nvec; idx++)
          if (vt[idx] == 0 && va[idx] == l1_addrs[k] && !found) begin
            read_and_check(0, va[idx], vh[idx]); found = 1;
          end
        if (!found) $fatal(1, "directed L1 address %0d missing from vectors", l1_addrs[k]);
      end
      for (k = 0; k < 6; k++) begin
        read_and_check(2, l2_addrs[k], {16'b0, l2w_exp[l2_addrs[k]]});
      end
    end

    // ---- all vectors, one read each ----
    for (k = 0; k < nvec; k++) read_and_check(vt[k], va[k], vh[k]);

    // ---- signedness corner cases ----
    expect_s(0, 0, 32'h0000ffff, -1);
    expect_s(0, 1, 32'h00008000, -32768);
    expect_s(0, 2, 32'h00007fff, 32767);
    expect_s(0, 3, 32'h00000001, 1);
    expect_s(0, 4, 32'h00000000, 0);
    expect_s(0, 5, 32'h0000fffe, -2);
    expect_s(0, 6, 32'h00008001, -32767);
    expect_s(1, 0, 32'hffffffff, -1);
    expect_s(1, 1, 32'h80000000, -64'sd2147483648);
    expect_s(1, 2, 32'h7fffffff, 2147483647);
    expect_s(1, 3, 32'h00000001, 1);
    expect_s(1, 4, 32'h00000000, 0);
    expect_s(1, 5, 32'hffff0000, -65536);

    // ---- latency: data lags the address by exactly one clock ----
    // Preload A, then change to B and check the output is still A until the next edge.
    if (!EARLY) begin
      int A, B;
      A = 5; B = 6;                         // l2w_exp[5] != l2w_exp[6] is asserted below
      if (l2w_exp[A] === l2w_exp[B]) begin A = 0; B = 1; end
      a_l2w = A[L2_WEIGHT_ADDR_WIDTH-1:0];
      @(posedge clk); #1;
      latency_tests++; if (d_l2w !== l2w_exp[A]) latency_fails++;      // after edge: mem[A]
      a_l2w = B[L2_WEIGHT_ADDR_WIDTH-1:0];
      #3;
      latency_tests++; if (d_l2w !== l2w_exp[A]) latency_fails++;      // still A before next edge
      @(posedge clk); #1;
      latency_tests++; if (d_l2w !== l2w_exp[B]) latency_fails++;      // now B
      // data must not follow addr combinationally
      latency_tests++; if (l2w_exp[A] === l2w_exp[B]) latency_fails++;

      // back-to-back stream: a new address every clock, data lags by one cycle
      a_l2w = '0;
      @(posedge clk); #1;
      for (k = 1; k < L2_WEIGHT_DEPTH; k++) begin
        a_l2w = k[L2_WEIGHT_ADDR_WIDTH-1:0];
        #3;
        stream_tests++; if (d_l2w !== l2w_exp[k-1]) stream_fails++;   // old data mid-cycle
        @(posedge clk); #1;
        stream_tests++; if (d_l2w !== l2w_exp[k]) stream_fails++;     // new data after edge
      end
    end

    total_fail = signed_fails + latency_fails + stream_fails;
    for (k = 0; k < 4; k++) total_fail += fails[k];

    $display("----------------------------------------------");
    $display("L1 weight tests : %0d  mismatches: %0d", tests[0], fails[0]);
    $display("L1 bias tests   : %0d  mismatches: %0d", tests[1], fails[1]);
    $display("L2 weight tests : %0d  mismatches: %0d", tests[2], fails[2]);
    $display("L2 bias tests   : %0d  mismatches: %0d", tests[3], fails[3]);
    $display("negative parameters checked as negative: %0d", neg_checked);
    $display("signed corner tests : %0d  failures: %0d", signed_tests, signed_fails);
    $display("latency tests       : %0d  failures: %0d", latency_tests, latency_fails);
    $display("back-to-back checks : %0d  failures: %0d", stream_tests, stream_fails);
    $display("total failures: %0d", total_fail);
    if (total_fail == 0) $display("PASS: tb_parameter_roms");
    else $fatal(1, "FAIL: tb_parameter_roms (%0d failures)", total_fail);
    $finish;
  end
endmodule
