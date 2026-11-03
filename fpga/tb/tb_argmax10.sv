// Self-checking testbench for argmax10 (RTL milestone 11).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/argmax10.sv ../rtl/output_spike_counters.sv tb_argmax10.sv
//   vsim -c work.tb_argmax10 -do "run -all; quit -f"
//
// Part 1: vectors from vectors/argmax_vec.txt (22 real Int16SNN counts, directed, 50,000 random),
//         oracle np.argmax. Counts are driven, the combinational result is checked after #1.
// Part 2: output_spike_counters -> argmax10: for the 22 images 25 count pulses of the real
//         output spikes (vectors/oc_steps.txt), then the prediction is compared with
//         vectors/oc_final.txt (Int16SNN). The prediction is only sampled after the 25th pulse.
// Part 3: direct tie-semantics checks (counts[2] = counts[5] = 9 -> 2, all zero -> 0).
`timescale 1ns/1ps

module tb_argmax10;
  import snn_params_pkg::*;

  localparam int NREAL = 22;
  localparam int NDIR  = 84;
  localparam int NRAND = 50000;
  localparam int NVEC  = NREAL + NDIR + NRAND;
  localparam int NSTEP = NUM_STEPS;

  logic clk = 1'b0;
  always #5 clk = ~clk;

  // ---------------- DUT driven directly ----------------
  logic [SPIKE_COUNT_WIDTH-1:0] counts [0:OUTPUT_SIZE-1];
  logic [3:0]                   pred;
  logic [SPIKE_COUNT_WIDTH-1:0] maxc;
  argmax10 dut (.counts(counts), .predicted_digit(pred), .max_count(maxc));

  // ---------------- counters + argmax chain ----------------
  logic rst, clr, en;
  logic [OUTPUT_SIZE-1:0] spk;
  logic [SPIKE_COUNT_WIDTH-1:0] ccounts [0:OUTPUT_SIZE-1];
  logic [3:0] cpred;
  logic [SPIKE_COUNT_WIDTH-1:0] cmax;
  output_spike_counters cnt (.clk(clk), .rst(rst), .clear_counts(clr), .count_enable(en),
                             .output_spikes(spk), .counts(ccounts));
  argmax10 amx (.counts(ccounts), .predicted_digit(cpred), .max_count(cmax));

  int vc [0:NVEC*OUTPUT_SIZE-1];
  int vp [0:NVEC-1];
  int vm [0:NVEC-1];
  int stspk [0:NREAL*NSTEP-1];
  int fpred [0:NREAL-1];
  int fcnt  [0:NREAL*OUTPUT_SIZE-1];
  int v [0:11];
  int fd, rc;

  int n_real, n_dir, n_rand, n_tie, comps, fails;
  int real_fail, dir_fail, rand_fail, tie_fail, max_fail, tie_vectors_checked;
  int int_images, int_pulses, int_pred_cmp, int_fail, int_cnt_cmp, int_cnt_fail, int_max_fail;
  int allzero_fail, special_fail, special_checks;
  int exp_pred;

  function automatic bit is_tie(input int idx);
    int m = 0, n = 0;
    for (int k = 0; k < OUTPUT_SIZE; k++) if (vc[idx*OUTPUT_SIZE + k] > m) m = vc[idx*OUTPUT_SIZE + k];
    for (int k = 0; k < OUTPUT_SIZE; k++) if (vc[idx*OUTPUT_SIZE + k] == m) n++;
    return n > 1;
  endfunction

  task automatic show(input int idx);
    $display("  counts=%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d exp_pred=%0d got_pred=%0d exp_max=%0d got_max=%0d",
             vc[idx*10], vc[idx*10+1], vc[idx*10+2], vc[idx*10+3], vc[idx*10+4], vc[idx*10+5], vc[idx*10+6],
             vc[idx*10+7], vc[idx*10+8], vc[idx*10+9], vp[idx], pred, vm[idx], maxc);
  endtask

  task automatic run_vec(input int idx);
    bit bad;
    for (int k = 0; k < OUTPUT_SIZE; k++) counts[k] = SPIKE_COUNT_WIDTH'(vc[idx*OUTPUT_SIZE + k]);
    #1;
    comps += 2;
    bad = 1'b0;
    if (int'(pred) !== vp[idx]) begin
      bad = 1'b1;
      if (idx < NREAL) real_fail++; else if (idx < NREAL + NDIR) dir_fail++; else rand_fail++;
      if (is_tie(idx)) tie_fail++;
    end
    if (int'(maxc) !== vm[idx]) begin bad = 1'b1; max_fail++; end
    if (bad && (real_fail + dir_fail + rand_fail + max_fail) <= 10) begin
      $display("ARGMAX FAIL vector %0d", idx); show(idx);
    end
    if (is_tie(idx)) tie_vectors_checked++;
    if (idx < NREAL) n_real++; else if (idx < NREAL + NDIR) n_dir++; else n_rand++;
    #1;
  endtask

  task automatic check_special(input int c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, input int expp, input string what);
    counts[0] = SPIKE_COUNT_WIDTH'(c0); counts[1] = SPIKE_COUNT_WIDTH'(c1); counts[2] = SPIKE_COUNT_WIDTH'(c2);
    counts[3] = SPIKE_COUNT_WIDTH'(c3); counts[4] = SPIKE_COUNT_WIDTH'(c4); counts[5] = SPIKE_COUNT_WIDTH'(c5);
    counts[6] = SPIKE_COUNT_WIDTH'(c6); counts[7] = SPIKE_COUNT_WIDTH'(c7); counts[8] = SPIKE_COUNT_WIDTH'(c8);
    counts[9] = SPIKE_COUNT_WIDTH'(c9);
    #1;
    special_checks++;
    if (int'(pred) !== expp) begin special_fail++; $display("SPECIAL FAIL %s: got=%0d exp=%0d", what, pred, expp); end
    #1;
  endtask

  task automatic cstep(input logic [OUTPUT_SIZE-1:0] s, input logic e, input logic c);
    spk = s; en = e; clr = c;
    @(posedge clk); #1;
  endtask

  task automatic run_integration();
    for (int im = 0; im < NREAL; im++) begin
      cstep('0, 1'b0, 1'b1);
      for (int t = 0; t < NSTEP; t++) begin
        cstep(stspk[im*NSTEP + t][OUTPUT_SIZE-1:0], 1'b1, 1'b0);
        int_pulses++;
      end
      cstep('0, 1'b0, 1'b0);          // pulses complete; prediction is read now
      for (int k = 0; k < OUTPUT_SIZE; k++) begin
        int_cnt_cmp++;
        if (int'(ccounts[k]) !== fcnt[im*OUTPUT_SIZE + k]) begin
          int_cnt_fail++; $display("INT COUNT FAIL img=%0d k=%0d got=%0d exp=%0d", im, k, ccounts[k], fcnt[im*OUTPUT_SIZE + k]);
        end
      end
      int_pred_cmp++;
      if (int'(cpred) !== fpred[im]) begin
        int_fail++; $display("INT PRED FAIL img=%0d got=%0d exp=%0d", im, cpred, fpred[im]);
      end
      if (int'(cmax) !== vm[im]) begin
        int_max_fail++; $display("INT MAX FAIL img=%0d got=%0d exp=%0d", im, cmax, vm[im]);
      end
      int_images++;
    end
  endtask

  initial begin
    n_real = 0; n_dir = 0; n_rand = 0; n_tie = 0; comps = 0; fails = 0;
    real_fail = 0; dir_fail = 0; rand_fail = 0; tie_fail = 0; max_fail = 0; tie_vectors_checked = 0;
    int_images = 0; int_pulses = 0; int_pred_cmp = 0; int_fail = 0; int_cnt_cmp = 0; int_cnt_fail = 0; int_max_fail = 0;
    allzero_fail = 0; special_fail = 0; special_checks = 0;
    rst = 1'b1; clr = 1'b0; en = 1'b0; spk = '0;
    for (int k = 0; k < OUTPUT_SIZE; k++) counts[k] = '0;

    fd = $fopen("vectors/argmax_vec.txt", "r");
    if (fd == 0) begin $display("cannot open argmax_vec.txt"); $finish; end
    for (int i = 0; i < NVEC; i++) begin
      rc = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h %h", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
      if (rc != 12) begin $display("argmax_vec read error line %0d rc=%0d", i, rc); $finish; end
      for (int k = 0; k < OUTPUT_SIZE; k++) vc[i*OUTPUT_SIZE + k] = v[k];
      vp[i] = v[10]; vm[i] = v[11];
    end
    $fclose(fd);
    fd = $fopen("vectors/oc_steps.txt", "r");
    for (int i = 0; i < NREAL*NSTEP; i++) begin
      rc = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
      if (rc != 11) begin $display("oc_steps read error line %0d", i); $finish; end
      stspk[i] = v[0];
    end
    $fclose(fd);
    fd = $fopen("vectors/oc_final.txt", "r");
    for (int i = 0; i < NREAL; i++) begin
      rc = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
      if (rc != 11) begin $display("oc_final read error line %0d", i); $finish; end
      fpred[i] = v[0];
      for (int k = 0; k < OUTPUT_SIZE; k++) fcnt[i*OUTPUT_SIZE + k] = v[k+1];
    end
    $fclose(fd);

    #10;
    for (int i = 0; i < NVEC; i++) run_vec(i);

    // direct tie semantics
    check_special(0,0,9,0,0,9,0,0,0,0, 2, "tie 2/5 (9)");
    check_special(0,0,9,6,0,9,0,6,0,0, 2, "all-255 counts");
    check_special(0,2,5,1,3,5,0,1,0,0, 2, "spec example");
    check_special(0,0,0,0,0,0,0,0,0,0, 0, "all zero");
    check_special(5,0,0,0,0,0,0,0,0,5, 0, "tie 0/9");
    check_special(0,5,0,0,5,0,0,0,5,0, 1, "three-way 1,4,8");
    check_special(7,7,7,7,7,7,7,7,7,7, 0, "all equal 7");
    check_special(0,0,0,0,0,0,0,0,0,25, 9, "unique max at 9");
    allzero_fail = special_fail;

    // counters + argmax
    #10 rst = 1'b0;
    @(posedge clk); #1;
    run_integration();

    fails = real_fail + dir_fail + rand_fail + max_fail + special_fail + int_fail + int_cnt_fail + int_max_fail;
    $display("---------------- RESULTS ----------------");
    $display("real image count vectors=%0d failures=%0d", n_real, real_fail);
    $display("directed vectors=%0d failures=%0d", n_dir, dir_fail);
    $display("random vectors=%0d failures=%0d", n_rand, rand_fail);
    $display("vectors with a tie for the maximum=%0d, tie failures=%0d", tie_vectors_checked, tie_fail);
    $display("special tie/all-zero checks=%0d failures=%0d", special_checks, special_fail);
    $display("total comparisons (prediction + max_count, vectors)=%0d, max_count failures=%0d", comps, max_fail);
    $display("integration: images=%0d count pulses=%0d prediction comparisons=%0d failures=%0d, final count comparisons=%0d failures=%0d, max failures=%0d",
             int_images, int_pulses, int_pred_cmp, int_fail, int_cnt_cmp, int_cnt_fail, int_max_fail);
    $display("total failures=%0d", fails);
    if (fails == 0) $display("PASS tb_argmax10");
    else            $display("FAIL tb_argmax10");
    $finish;
  end

  initial begin
    #500000000;
    $display("TIMEOUT");
    $finish;
  end
endmodule
