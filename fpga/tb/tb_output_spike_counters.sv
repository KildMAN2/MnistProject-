// Self-checking testbench for output_spike_counters (RTL milestone 10).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv ../rtl/l2_acc.sv \
//            ../rtl/l2_current_engine.sv ../rtl/decay_19_20.sv ../rtl/lif_core.sv \
//            ../rtl/l2_lif_sweep.sv ../rtl/output_spike_counters.sv tb_output_spike_counters.sv
//   vsim -c work.tb_output_spike_counters -do "run -all; quit -f"
//
// Part 1 (dut): standalone counter bank, driven directly. A behavioural model (exp[]) is
//               updated for every clock; golden cumulative Python counts are compared too.
// Part 2 (icnt): the same RTL behind the real l2_current_engine + l2_lif_sweep; one
//               count_enable pulse after every step_done.
// Stimulus 1 ns after posedge; results are compared #1 after the next posedge, i.e. after
// the counter registers have updated.
`timescale 1ns/1ps

module tb_output_spike_counters;
  import snn_params_pkg::*;

  localparam int NIMG  = 22;
  localparam int NSTEP = NUM_STEPS;
  localparam int NINT  = 6;
  localparam int N_PULSES_ALL_ONE = NSTEP;    // intentional-break hook: 24 / 26

  logic clk = 1'b0;
  always #5 clk = ~clk;
  logic rst;

  // ---------------- DUT (standalone) ----------------
  logic                         d_clr, d_en;
  logic [OUTPUT_SIZE-1:0]       d_spk;
  logic [SPIKE_COUNT_WIDTH-1:0] counts [0:OUTPUT_SIZE-1];

  output_spike_counters dut (.clk(clk), .rst(rst), .clear_counts(d_clr), .count_enable(d_en),
                             .output_spikes(d_spk), .counts(counts));

  // ---------------- integration chain ----------------
  logic        e_start, e_busy, e_done;
  logic [HIDDEN_SIZE-1:0] hidden_spikes;
  logic signed [L2_ACC_WIDTH-1:0] cur2 [0:OUTPUT_SIZE-1];
  l2_current_engine eng (.clk(clk), .rst(rst), .start(e_start), .hidden_spikes(hidden_spikes),
                         .busy(e_busy), .done(e_done), .cur2(cur2));

  logic clear_state, clear_done, start_step, busy, step_done;
  logic [OUTPUT_SIZE-1:0] output_spikes;
  l2_lif_sweep sweep (.clk(clk), .rst(rst), .clear_state(clear_state), .clear_done(clear_done),
                      .start_step(start_step), .busy(busy), .step_done(step_done),
                      .cur2(cur2), .output_spikes(output_spikes));

  logic i_clr, i_en;
  logic [SPIKE_COUNT_WIDTH-1:0] icounts [0:OUTPUT_SIZE-1];
  output_spike_counters icnt (.clk(clk), .rst(rst), .clear_counts(i_clr), .count_enable(i_en),
                              .output_spikes(output_spikes), .counts(icounts));

  // ---------------- vectors ----------------
  logic [255:0] hsm [0:NINT*NSTEP-1];
  int           stspk [0:NIMG*NSTEP-1];
  int           stc   [0:NIMG*NSTEP*OUTPUT_SIZE-1];
  int           fpred [0:NIMG-1];
  int           fcnt  [0:NIMG*OUTPUT_SIZE-1];
  int           fd, rc;
  int           v [0:10];
  int           int_img [0:NINT-1] = '{0, 1, 2, 3, 20, 21};

  // ---------------- model and statistics ----------------
  int exp_cnt [0:OUTPUT_SIZE-1];
  int cmp_total, cmp_fail, pulses, clk_cycles;
  int gold_cmp, gold_fail, gold_steps, gold_images, gold_pulses;
  int dir_tests, dir_fail;
  int clr_tests, clr_fail, hold_tests, hold_fail, rst_tests, rst_fail, prio_tests, prio_fail;
  int allone_fail, max_count, rand_cycles;
  int int_images, int_steps, int_cmp, int_fail, int_pulses, int_final_cmp, int_final_fail, sd_pulses;
  int fails;

  function automatic int model_mask();
    return (1 << SPIKE_COUNT_WIDTH) - 1;
  endfunction

  // One clock: apply inputs, update model, compare all ten counters against the model.
  task automatic cyc(input logic [OUTPUT_SIZE-1:0] spk, input logic en, input logic clr);
    d_spk = spk; d_en = en; d_clr = clr;
    @(posedge clk); #1;
    clk_cycles++;
    if (clr) begin
      for (int k = 0; k < OUTPUT_SIZE; k++) exp_cnt[k] = 0;
    end else if (en) begin
      pulses++;
      for (int k = 0; k < OUTPUT_SIZE; k++) if (spk[k]) exp_cnt[k] = (exp_cnt[k] + 1) & model_mask();
    end
    for (int k = 0; k < OUTPUT_SIZE; k++) begin
      cmp_total++;
      if (counts[k] !== SPIKE_COUNT_WIDTH'(exp_cnt[k])) begin
        cmp_fail++;
        if (cmp_fail <= 10) $display("MODEL FAIL k=%0d spk=%b en=%b clr=%b got=%0d exp=%0d", k, spk, en, clr, counts[k], exp_cnt[k]);
      end
    end
  endtask

  task automatic idle();
    d_spk = '0; d_en = 1'b0; d_clr = 1'b0;
  endtask

  task automatic do_reset();
    rst = 1'b1; d_clr = 1'b0; d_en = 1'b0; d_spk = '0; i_clr = 1'b0; i_en = 1'b0;
    e_start = 1'b0; clear_state = 1'b0; start_step = 1'b0; hidden_spikes = '0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    for (int k = 0; k < OUTPUT_SIZE; k++) exp_cnt[k] = 0;
    @(posedge clk); #1;
  endtask

  task automatic expect_all(input int val, input string what);
    for (int k = 0; k < OUTPUT_SIZE; k++) begin
      cmp_total++;
      if (counts[k] !== SPIKE_COUNT_WIDTH'(val)) begin
        cmp_fail++;
        $display("%s FAIL k=%0d got=%0d exp=%0d", what, k, counts[k], val);
      end
    end
  endtask

  // ---------------- part 1a: golden cumulative counts ----------------
  task automatic run_golden();
    for (int im = 0; im < NIMG; im++) begin
      cyc('0, 1'b0, 1'b1);                      // clear
      expect_all(0, "GOLDEN CLEAR");
      for (int t = 0; t < NSTEP; t++) begin
        cyc(stspk[im*NSTEP + t][OUTPUT_SIZE-1:0], 1'b1, 1'b0);
        gold_steps++; gold_pulses++;
        for (int k = 0; k < OUTPUT_SIZE; k++) begin
          gold_cmp++;
          if (int'(counts[k]) !== stc[(im*NSTEP + t)*OUTPUT_SIZE + k]) begin
            gold_fail++;
            if (gold_fail <= 10) $display("GOLDEN FAIL img=%0d t=%0d k=%0d got=%0d exp=%0d", im, t, k, counts[k],
                                          stc[(im*NSTEP + t)*OUTPUT_SIZE + k]);
          end
          if (int'(counts[k]) > max_count) max_count = int'(counts[k]);
        end
      end
      for (int k = 0; k < OUTPUT_SIZE; k++) begin     // final vector
        gold_cmp++;
        if (int'(counts[k]) !== fcnt[im*OUTPUT_SIZE + k]) begin
          gold_fail++; $display("GOLDEN FINAL FAIL img=%0d k=%0d got=%0d exp=%0d", im, k, counts[k], fcnt[im*OUTPUT_SIZE + k]);
        end
      end
      gold_images++;
    end
    idle();
  endtask

  // ---------------- part 1b: directed ----------------
  task automatic check_delta(input int prev [0:OUTPUT_SIZE-1], input logic [OUTPUT_SIZE-1:0] spk, input string what);
    dir_tests++;
    for (int k = 0; k < OUTPUT_SIZE; k++) begin
      cmp_total++;
      if (int'(counts[k]) !== prev[k] + (spk[k] ? 1 : 0)) begin
        dir_fail++; cmp_fail++;
        $display("%s FAIL k=%0d got=%0d exp=%0d", what, k, counts[k], prev[k] + (spk[k] ? 1 : 0));
      end
    end
  endtask

  task automatic snap(output int s [0:OUTPUT_SIZE-1]);
    for (int k = 0; k < OUTPUT_SIZE; k++) s[k] = int'(counts[k]);
  endtask

  task automatic single_pulse(input logic [OUTPUT_SIZE-1:0] spk, input string what);
    int prev [0:OUTPUT_SIZE-1];
    snap(prev);
    cyc(spk, 1'b1, 1'b0);
    check_delta(prev, spk, what);
  endtask

  task automatic run_directed();
    int prev [0:OUTPUT_SIZE-1];
    logic [OUTPUT_SIZE-1:0] r;
    cyc('0, 1'b0, 1'b1);
    single_pulse(10'b0000000000, "ALLZERO");
    single_pulse(10'b1111111111, "ALLONE");
    single_pulse(10'b0000000001, "SINGLE0");
    single_pulse(10'b1000000000, "SINGLE9");
    for (int k = 0; k < OUTPUT_SIZE; k++) single_pulse(OUTPUT_SIZE'(1) << k, "WALKING");
    single_pulse(10'b0101010101, "ALT_A");
    single_pulse(10'b1010101010, "ALT_B");
    single_pulse(10'b0000000101, "EXAMPLE_0_2");
    for (int i = 0; i < 200; i++) begin r = $urandom; single_pulse(r, "RANDOM_PULSE"); cyc('0, 1'b0, 1'b1); end

    // spike vector held for five clocks, count_enable high for one clock only
    cyc('0, 1'b0, 1'b1);
    snap(prev);
    cyc(10'b1010011101, 1'b1, 1'b0);
    check_delta(prev, 10'b1010011101, "HOLD_FIRST");
    snap(prev);
    for (int i = 0; i < 4; i++) begin
      cyc(10'b1010011101, 1'b0, 1'b0);
      hold_tests++;
      for (int k = 0; k < OUTPUT_SIZE; k++) if (int'(counts[k]) !== prev[k]) begin hold_fail++; cmp_fail++; $display("HOLD FAIL k=%0d", k); end
    end
    // enable low with all-ones for many clocks: nothing changes
    snap(prev);
    for (int i = 0; i < 20; i++) begin
      cyc(10'b1111111111, 1'b0, 1'b0);
      hold_tests++;
      for (int k = 0; k < OUTPUT_SIZE; k++) if (int'(counts[k]) !== prev[k]) begin hold_fail++; cmp_fail++; $display("HOLD FAIL k=%0d", k); end
    end

    // exactly 25 all-one pulses
    cyc('0, 1'b0, 1'b1);
    for (int i = 0; i < N_PULSES_ALL_ONE; i++) begin
      cyc(10'b1111111111, 1'b1, 1'b0);
      for (int k = 0; k < OUTPUT_SIZE; k++) if (int'(counts[k]) > max_count) max_count = int'(counts[k]);
    end
    for (int k = 0; k < OUTPUT_SIZE; k++) begin
      dir_tests++;
      if (counts[k] !== SPIKE_COUNT_WIDTH'(NUM_STEPS)) begin
        allone_fail++; dir_fail++; cmp_fail++;
        $display("25-PULSE FAIL k=%0d got=%0d exp=%0d", k, counts[k], NUM_STEPS);
      end
    end
    idle();
  endtask

  // ---------------- part 1c: clear tests ----------------
  task automatic run_clear_tests();
    logic [OUTPUT_SIZE-1:0] r;
    // clear in the middle of an image, then continue from zero
    cyc('0, 1'b0, 1'b1);
    for (int i = 0; i < 10; i++) begin r = $urandom | 10'b0000100000; cyc(r, 1'b1, 1'b0); end
    clr_tests++;
    for (int k = 0; k < OUTPUT_SIZE; k++) if (exp_cnt[k] == 0 && k == 5) begin clr_fail++; $display("pre-clear counter 5 is zero"); end
    cyc(10'b1111111111, 1'b0, 1'b1);
    clr_tests++;
    expect_all(0, "MID CLEAR");
    for (int i = 0; i < 5; i++) cyc(10'b0000100001, 1'b1, 1'b0);
    clr_tests++;
    if (counts[0] !== 5'd5 || counts[5] !== 5'd5 || counts[1] !== 5'd0) begin clr_fail++; cmp_fail++; $display("CONTINUE FAIL"); end

    // priority: clear and enable together with all ones
    for (int i = 0; i < 3; i++) begin
      cyc(10'b1111111111, 1'b1, 1'b1);
      prio_tests++;
      for (int k = 0; k < OUTPUT_SIZE; k++) if (counts[k] !== '0) begin prio_fail++; cmp_fail++; $display("PRIORITY FAIL k=%0d got=%0d", k, counts[k]); end
    end
    // priority with non-zero counters and a single spike
    for (int i = 0; i < 4; i++) cyc(10'b1111111111, 1'b1, 1'b0);
    cyc(10'b0000100000, 1'b1, 1'b1);
    prio_tests++;
    for (int k = 0; k < OUTPUT_SIZE; k++) if (counts[k] !== '0) begin prio_fail++; cmp_fail++; $display("PRIORITY2 FAIL k=%0d", k); end

    // clear with all counters non-zero (each counter individually must clear)
    for (int i = 0; i < 7; i++) cyc(10'b1111111111, 1'b1, 1'b0);
    cyc('0, 1'b0, 1'b1);
    clr_tests++;
    expect_all(0, "CLEAR ALL");
    idle();
  endtask

  // ---------------- part 1d: reset ----------------
  task automatic run_reset_tests();
    cyc('0, 1'b0, 1'b0);
    expect_all(0, "RESET INITIAL");
    for (int i = 0; i < 9; i++) cyc(10'b1111111111, 1'b1, 1'b0);
    for (int i = 0; i < 3; i++) begin
      d_spk = 10'h3FF; d_en = 1'b1;
      rst = 1'b1;
      @(posedge clk); #1;
      rst = 1'b0;
      for (int k = 0; k < OUTPUT_SIZE; k++) exp_cnt[k] = 0;
      d_en = 1'b0;
      rst_tests++;
      for (int k = 0; k < OUTPUT_SIZE; k++) if (counts[k] !== '0) begin rst_fail++; cmp_fail++; $display("RESET FAIL k=%0d got=%0d", k, counts[k]); end
      cyc(10'b1111111111, 1'b1, 1'b0);
      cyc(10'b1111111111, 1'b1, 1'b0);
    end
    cyc('0, 1'b0, 1'b1);
    idle();
  endtask

  // ---------------- part 1e: random with random clear/enable ----------------
  task automatic run_random();
    logic [OUTPUT_SIZE-1:0] r;
    for (int i = 0; i < 3000; i++) begin
      r = $urandom;
      cyc(r, ($urandom % 2) == 0, ($urandom % 16) == 0);
      rand_cycles++;
    end
    cyc('0, 1'b0, 1'b1);
    idle();
  endtask

  // ---------------- part 2: integration ----------------
  task automatic run_engine(input int hidx);
    int cyc_n;
    hidden_spikes = hsm[hidx];
    e_start = 1'b1;
    @(posedge clk); #1 e_start = 1'b0;
    cyc_n = 0;
    while (!e_done && cyc_n < 5000) begin @(posedge clk); #1; cyc_n++; end
    if (cyc_n >= 5000) begin int_fail++; $display("ENGINE timeout"); end
    @(posedge clk); #1;
  endtask

  task automatic run_integration();
    for (int ii = 0; ii < NINT; ii++) begin
      int im = int_img[ii];
      // clear Layer-2 state and the counters
      clear_state = 1'b1; i_clr = 1'b1;
      @(posedge clk); #1 clear_state = 1'b0; i_clr = 1'b0;
      while (!clear_done) begin @(posedge clk); #1; end
      for (int k = 0; k < OUTPUT_SIZE; k++) begin
        int_cmp++;
        if (icounts[k] !== '0) begin int_fail++; $display("INT clear FAIL k=%0d", k); end
      end
      for (int t = 0; t < NSTEP; t++) begin
        int cn;
        run_engine(ii*NSTEP + t);
        start_step = 1'b1;
        @(posedge clk); #1 start_step = 1'b0;
        cn = 0;
        while (!step_done && cn < 100) begin @(posedge clk); #1; cn++; end
        sd_pulses++;
        // exactly one count_enable pulse for the completed step; step_done is NOT used as a level
        i_en = 1'b1;
        @(posedge clk); #1 i_en = 1'b0;
        int_pulses++;
        int_steps++;
        for (int k = 0; k < OUTPUT_SIZE; k++) begin
          int_cmp++;
          if (int'(icounts[k]) !== stc[(im*NSTEP + t)*OUTPUT_SIZE + k]) begin
            int_fail++;
            if (int_fail <= 10) $display("INT FAIL img=%0d t=%0d k=%0d got=%0d exp=%0d", im, t, k, icounts[k],
                                         stc[(im*NSTEP + t)*OUTPUT_SIZE + k]);
          end
        end
        // spikes unchanged and counters unchanged for several idle clocks (no level counting)
        repeat (3) begin
          @(posedge clk); #1;
          for (int k = 0; k < OUTPUT_SIZE; k++) begin
            int_cmp++;
            if (int'(icounts[k]) !== stc[(im*NSTEP + t)*OUTPUT_SIZE + k]) begin int_fail++; $display("INT level-count FAIL"); end
          end
        end
      end
      for (int k = 0; k < OUTPUT_SIZE; k++) begin
        int_final_cmp++;
        if (int'(icounts[k]) !== fcnt[im*OUTPUT_SIZE + k]) begin
          int_final_fail++; $display("INT FINAL FAIL img=%0d k=%0d got=%0d exp=%0d", im, k, icounts[k], fcnt[im*OUTPUT_SIZE + k]);
        end
      end
      int_images++;
    end
  endtask

  // ---------------- main ----------------
  initial begin
    cmp_total = 0; cmp_fail = 0; pulses = 0; clk_cycles = 0;
    gold_cmp = 0; gold_fail = 0; gold_steps = 0; gold_images = 0; gold_pulses = 0;
    dir_tests = 0; dir_fail = 0; clr_tests = 0; clr_fail = 0; hold_tests = 0; hold_fail = 0;
    rst_tests = 0; rst_fail = 0; prio_tests = 0; prio_fail = 0; allone_fail = 0; max_count = 0; rand_cycles = 0;
    int_images = 0; int_steps = 0; int_cmp = 0; int_fail = 0; int_pulses = 0; int_final_cmp = 0; int_final_fail = 0; sd_pulses = 0;

    fd = $fopen("vectors/oc_steps.txt", "r");
    if (fd == 0) begin $display("cannot open oc_steps.txt"); $finish; end
    for (int i = 0; i < NIMG*NSTEP; i++) begin
      rc = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
      if (rc != 11) begin $display("oc_steps read error line %0d rc=%0d", i, rc); $finish; end
      stspk[i] = v[0];
      for (int k = 0; k < OUTPUT_SIZE; k++) stc[i*OUTPUT_SIZE + k] = v[k+1];
    end
    $fclose(fd);
    fd = $fopen("vectors/oc_final.txt", "r");
    if (fd == 0) begin $display("cannot open oc_final.txt"); $finish; end
    for (int i = 0; i < NIMG; i++) begin
      rc = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
      if (rc != 11) begin $display("oc_final read error line %0d rc=%0d", i, rc); $finish; end
      fpred[i] = v[0];
      for (int k = 0; k < OUTPUT_SIZE; k++) fcnt[i*OUTPUT_SIZE + k] = v[k+1];
    end
    $fclose(fd);
    $readmemh("vectors/oc_hs.txt", hsm);

    do_reset();
    cyc('0, 1'b0, 1'b0);
    expect_all(0, "RESET");
    rst_tests++;
    run_golden();
    run_directed();
    run_clear_tests();
    run_reset_tests();
    run_random();
    do_reset();
    run_integration();

    fails = cmp_fail + gold_fail + dir_fail + clr_fail + hold_fail + rst_fail + prio_fail + allone_fail +
            int_fail + int_final_fail;
    $display("---------------- RESULTS ----------------");
    $display("golden: images=%0d time_steps=%0d count_enable pulses=%0d counter comparisons=%0d (incl. final) failures=%0d",
             gold_images, gold_steps, gold_pulses, gold_cmp, gold_fail);
    $display("model-compared counter values (all phases)=%0d failures=%0d, count_enable pulses (all phases)=%0d, clocks=%0d",
             cmp_total, cmp_fail, pulses, clk_cycles);
    $display("directed checks=%0d failures=%0d (all-one x%0d pulses failures=%0d)", dir_tests, dir_fail, N_PULSES_ALL_ONE, allone_fail);
    $display("hold (enable low) checks=%0d failures=%0d", hold_tests, hold_fail);
    $display("clear tests=%0d failures=%0d; priority tests=%0d failures=%0d; reset tests=%0d failures=%0d",
             clr_tests, clr_fail, prio_tests, prio_fail, rst_tests, rst_fail);
    $display("random cycles=%0d", rand_cycles);
    $display("maximum observed count (valid inference tests)=%0d", max_count);
    $display("integration: images=%0d time_steps=%0d step_done=%0d count pulses=%0d step comparisons=%0d failures=%0d final comparisons=%0d failures=%0d",
             int_images, int_steps, sd_pulses, int_pulses, int_cmp, int_fail, int_final_cmp, int_final_fail);
    $display("total failures=%0d", fails);
    if (fails == 0) $display("PASS tb_output_spike_counters");
    else            $display("FAIL tb_output_spike_counters");
    $finish;
  end

  initial begin
    #500000000;
    $display("TIMEOUT");
    $finish;
  end
endmodule
