// Self-checking testbench for l2_current_engine (RTL milestone 8).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv ../rtl/l2_acc.sv \
//            ../rtl/l2_current_engine.sv tb_l2_current_engine.sv
//   vsim -c work.tb_l2_current_engine -do "run -all; quit -f"
// Vectors 0..99 are the REAL layer-1 spike trajectories (image-major, t = 0..24)
// of MNIST 0, MNIST 1, all-zero, all-255; the rest are directed vectors.
// Stimulus 1 ns after posedge, monitors at negedge, results checked #1 after an edge.
`timescale 1ns/1ps

module tb_l2_current_engine;
  import snn_params_pkg::*;

  localparam int NREAL    = 100;
  localparam int NVEC     = 188;
  localparam int NIMG     = 4;
  localparam int NSTEP    = 25;
  localparam int EXP_LAT  = 2590;   // 10 * 259

  logic clk = 1'b0;
  always #5 clk = ~clk;

  logic rst, start, busy, done;
  logic [HIDDEN_SIZE-1:0] hidden_spikes;
  logic signed [L2_ACC_WIDTH-1:0] cur2 [0:OUTPUT_SIZE-1];

  l2_current_engine dut (.*);

  logic [255:0] hsm  [0:NVEC-1];
  logic [32:0]  curm [0:NVEC*10-1];
  logic [15:0]  w2m  [0:L2_WEIGHT_DEPTH-1];
  logic [31:0]  b2m  [0:L2_BIAS_DEPTH-1];

  // ---------------- counters ----------------
  int runs, fails;
  int cur_cmp, cur_fail;
  int ops, ops_fail;                 // input consumptions
  int addr_checks, addr_fail;        // weight address issue sequence
  int align_checks, align_fail;      // weight / spike alignment at every consume
  int bias_checks, bias_fail;
  int cnt_checks, cnt_fail;
  int lat_checks, lat_fail, done_fail, busy_fail, ign_checks, rst_fail;
  int min_lat, max_lat;

  // ---------------- monitor ----------------
  bit mon_en;
  int cur_vec;
  int cap_count, op_count, done_count;
  int bias_k;

  always @(negedge clk) begin
    if (mon_en) begin
      // address sequence: every issue of a weight address in LOAD/RUN must be k*256+j in order
      if (dut.state === 3'd2 || dut.state === 3'd3) begin   // S_LOAD / S_RUN
        addr_checks++;
        if (cap_count >= 2560 || dut.w_raddr !== 12'(cap_count) ||
            dut.kidx !== 4'(cap_count / 256) || dut.jidx !== 8'(cap_count % 256)) begin
          addr_fail++;
          if (addr_fail <= 10) $display("ADDR FAIL expected %0d got %0d (k=%0d j=%0d)", cap_count,
                                        dut.w_raddr, dut.kidx, dut.jidx);
        end
        cap_count++;
      end
      // bias address: BIAS state must address output bias_k
      if (dut.state === 3'd1) begin
        bias_checks++;
        if (dut.kidx !== 4'(bias_k)) begin
          bias_fail++;
          $display("BIAS ADDR FAIL expected k=%0d got %0d", bias_k, dut.kidx);
        end
        bias_k++;
      end
      // every consume: weight = weight2[k][j], spike = hidden_spikes[j] for the SAME j
      if (dut.v1 === 1'b1) begin
        align_checks++; ops++;
        if (op_count >= 2560 ||
            dut.w_data !== w2m[op_count] ||
            dut.spk_d !== hidden_spikes[op_count % 256]) begin
          align_fail++;
          if (align_fail <= 10)
            $display("ALIGN FAIL op=%0d (k=%0d j=%0d) weight=%h exp=%h spike=%b exp=%b", op_count,
                     op_count / 256, op_count % 256, dut.w_data,
                     (op_count < 2560) ? w2m[op_count] : 16'hxxxx, dut.spk_d,
                     hidden_spikes[op_count % 256]);
        end
        op_count++;
      end
      if (done) done_count++;
    end
  end

  // ---------------- helpers ----------------
  task automatic do_reset();
    rst = 1'b1; start = 1'b0; hidden_spikes = '0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    @(posedge clk); #1;
  endtask

  // spam: extra start pulses while busy (output 0, middle output, output 9)
  task automatic run_vec(input int v, input bit spam);
    int cyc, bad;
    bit seen;
    mon_en = 1'b0;
    cap_count = 0; op_count = 0; done_count = 0; bias_k = 0; cur_vec = v;
    hidden_spikes = hsm[v];
    start = 1'b1;
    @(posedge clk);                       // accepting edge E0
    mon_en = 1'b1;
    #1 start = 1'b0;
    cyc = 0; seen = 1'b0;
    while (!seen && cyc < 5000) begin
      start = spam && (cyc == 100 || cyc == 1300 || cyc == 2400 || cyc == 2589);
      @(posedge clk); #1; cyc++;
      if (spam && (cyc == 100 || cyc == 1300 || cyc == 2400)) ign_checks++;
      if (!done && busy !== 1'b1) busy_fail++;
      if (done) seen = 1'b1;
    end
    start = 1'b0;
    runs++;
    lat_checks++;
    if (!seen) begin fails++; $display("TIMEOUT vec=%0d", v); end
    else begin
      if (cyc != EXP_LAT) begin lat_fail++; $display("LATENCY %0d != %0d (vec %0d)", cyc, EXP_LAT, v); end
      if (v == 0 || cyc < min_lat) min_lat = cyc;
      if (v == 0 || cyc > max_lat) max_lat = cyc;
      if (busy !== 1'b0) busy_fail++;
    end
    // all ten currents must already be final when done is first visible
    bad = 0;
    for (int k = 0; k < OUTPUT_SIZE; k++) begin
      cur_cmp++;
      if (cur2[k] !== curm[v * 10 + k]) begin
        bad++; cur_fail++;
        if (cur_fail <= 10) $display("CUR2 FAIL vec=%0d k=%0d got=%0d exp=%0d", v, k, cur2[k], $signed(curm[v * 10 + k]));
      end
    end
    repeat (4) @(posedge clk); #1;
    cnt_checks++;
    if (op_count != 2560 || cap_count != 2560 || bias_k != 10 || done_count != 1) begin
      cnt_fail++;
      $display("COUNT FAIL vec=%0d ops=%0d issues=%0d biases=%0d done pulses=%0d", v, op_count, cap_count, bias_k, done_count);
    end
    mon_en = 1'b0;
  endtask

  task automatic reset_test();
    mon_en = 1'b0;
    hidden_spikes = hsm[0];
    start = 1'b1;
    @(posedge clk); #1 start = 1'b0;
    repeat (700) @(posedge clk);
    #1 rst = 1'b1;
    @(posedge clk); #1;
    if (busy !== 1'b0 || done !== 1'b0 || dut.state !== 3'd0) rst_fail++;
    if (dut.acc_out !== '0) rst_fail++;
    rst = 1'b0;
    repeat (3) @(posedge clk); #1;
    if (busy !== 1'b0) rst_fail++;
  endtask

  initial begin
    $readmemh("vectors/l2c_hs.txt", hsm);
    $readmemh("vectors/l2c_cur.txt", curm);
    $readmemh(L2_WEIGHT_FILE, w2m);
    $readmemh(L2_BIAS_FILE, b2m);
    mon_en = 1'b0;
    do_reset();

    // all 188 vectors; extra start pulses while busy on a few of them
    for (int v = 0; v < NVEC; v++) run_vec(v, (v == 0) || (v == 37) || (v == 101) || (v == 150));

    // reset in the middle of a run, then a complete correct run
    reset_test();
    run_vec(30, 1'b0);
    run_vec(NREAL + 1, 1'b0);   // all-ones after reset

    fails += cur_fail + ops_fail + addr_fail + align_fail + bias_fail + cnt_fail + lat_fail +
             done_fail + busy_fail + rst_fail;

    $display("engine runs                        : %0d", runs);
    $display("images / time steps (real vectors) : %0d / %0d (%0d real vectors)", NIMG, NSTEP, NREAL);
    $display("layer-2 current comparisons        : %0d", cur_cmp);
    $display("input-consumption operations       : %0d (%0d per run)", ops, 2560);
    $display("weight-address issue checks        : %0d", addr_checks);
    $display("spike/weight alignment checks      : %0d", align_checks);
    $display("bias-address checks                : %0d", bias_checks);
    $display("per-run count checks               : %0d", cnt_checks);
    $display("latency checks / measured min,max  : %0d / %0d, %0d (expected %0d)", lat_checks, min_lat, max_lat, EXP_LAT);
    $display("start-while-busy pulses            : %0d", ign_checks);
    $display("failures: cur=%0d addr=%0d align=%0d bias=%0d count=%0d lat=%0d busy=%0d rst=%0d",
             cur_fail, addr_fail, align_fail, bias_fail, cnt_fail, lat_fail, busy_fail, rst_fail);
    $display("total failures: %0d", fails);
    if (fails == 0) $display("PASS tb_l2_current_engine");
    else $fatal(1, "FAIL tb_l2_current_engine");
    $finish;
  end
endmodule
