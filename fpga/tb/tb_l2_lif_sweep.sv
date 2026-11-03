// Self-checking testbench for l2_lif_sweep (RTL milestone 9).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv ../rtl/l2_acc.sv \
//            ../rtl/l2_current_engine.sv ../rtl/decay_19_20.sv ../rtl/lif_core.sv \
//            ../rtl/l2_lif_sweep.sv tb_l2_lif_sweep.sv
//   vsim -c work.tb_l2_lif_sweep -do "run -all; quit -f"
//
// DUT A (dut)  : l2_lif_sweep fed with the REAL hardware cur2 of l2_current_engine (eng).
//                Per image: clear -> 25 x { engine(hidden_spikes[t]) -> start_step }.
// DUT B (dutb) : second l2_lif_sweep driven by a testbench cur2 array; mem2 is preloaded
//                hierarchically to run directed (membrane, current) cases (10 per step).
// Stimulus 1 ns after posedge, monitors at negedge, results checked #1 after an edge.
`timescale 1ns/1ps

module tb_l2_lif_sweep;
  import snn_params_pkg::*;

  localparam int NIMG      = 4;
  localparam int NSTEP     = 25;
  localparam int NDIR      = 1000;
  localparam int EXP_CLR   = 10;
  localparam int EXP_STEP  = 10;
  localparam int EXP_ENG   = 2590;

  logic clk = 1'b0;
  always #5 clk = ~clk;
  logic rst;

  // ---------------- real current engine ----------------
  logic        e_start, e_busy, e_done;
  logic [HIDDEN_SIZE-1:0] hidden_spikes;
  logic signed [L2_ACC_WIDTH-1:0] cur2 [0:OUTPUT_SIZE-1];

  l2_current_engine eng (.clk(clk), .rst(rst), .start(e_start), .hidden_spikes(hidden_spikes),
                         .busy(e_busy), .done(e_done), .cur2(cur2));

  // ---------------- DUT A ----------------
  logic clear_state, clear_done, start_step, busy, step_done;
  logic [OUTPUT_SIZE-1:0] output_spikes;

  l2_lif_sweep dut (.clk(clk), .rst(rst), .clear_state(clear_state), .clear_done(clear_done),
                    .start_step(start_step), .busy(busy), .step_done(step_done),
                    .cur2(cur2), .output_spikes(output_spikes));

  // ---------------- DUT B ----------------
  logic b_clear, b_clear_done, b_start, b_busy, b_step_done;
  logic [OUTPUT_SIZE-1:0] b_spikes;
  logic signed [L2_ACC_WIDTH-1:0] bcur [0:OUTPUT_SIZE-1];

  l2_lif_sweep dutb (.clk(clk), .rst(rst), .clear_state(b_clear), .clear_done(b_clear_done),
                     .start_step(b_start), .busy(b_busy), .step_done(b_step_done),
                     .cur2(bcur), .output_spikes(b_spikes));

  // ---------------- vectors ----------------
  logic [255:0] hsm  [0:NIMG*NSTEP-1];
  logic [35:0]  trm  [0:NIMG*NSTEP*10*4-1];   // 4 fields per line, see tidx()
  logic [35:0]  dirm [0:NDIR*4-1];

  // $readmemh cannot read a 4-field line into 4 words, so the files are read as text below.
  int fd, rc;

  // ---------------- counters ----------------
  int updates, mem_in_cmp, mem_out_cmp, spk_cmp, cur_cmp, order_checks;
  int mem_in_fail, mem_out_fail, spk_fail, cur_fail, order_fail;
  int wr_count_checks, wr_count_fail, stepdone_checks, stepdone_fail;
  int clr_wr_checks, clr_wr_fail, clr_zero_checks, clr_zero_fail;
  int lat_checks, lat_fail, busy_checks, busy_fail, ign_checks, rst_checks, rst_fail;
  int dir_checks, dir_fail, eng_runs, eng_fail, b_clear_checks, b_clear_fail;
  int steps_run, images_run, clears_run, fails;
  int min_clr, max_clr, min_step, max_step;

  function automatic int tidx(input int img, input int t, input int k, input int f);
    return (((img * NSTEP) + t) * 10 + k) * 4 + f;
  endfunction

  // ---------------- monitor ----------------
  bit  mon_en;
  int  ph;                    // 1 = clear, 2 = step
  int  m_img, m_step;
  int  upd_count, clr_count, sd_count, cd_count;

  always @(negedge clk) begin
    if (mon_en) begin
      if (ph == 2 && dut.state === 2'd2) begin       // S_UPD
        order_checks++;
        if (upd_count >= 10 || dut.kidx !== 4'(upd_count)) begin
          order_fail++;
          if (order_fail <= 10) $display("ORDER FAIL expected k=%0d got %0d", upd_count, dut.kidx);
        end else begin
          updates++; mem_in_cmp++; mem_out_cmp++; spk_cmp++; cur_cmp++;
          if (dut.mem2[upd_count] !== trm[tidx(m_img, m_step, upd_count, 1)]) begin
            mem_in_fail++;
            if (mem_in_fail <= 10) $display("MEM_IN FAIL img=%0d t=%0d k=%0d got=%h exp=%h", m_img, m_step, upd_count,
                                            dut.mem2[upd_count], trm[tidx(m_img, m_step, upd_count, 1)]);
          end
          if (dut.cur2[upd_count] !== trm[tidx(m_img, m_step, upd_count, 0)][32:0]) begin
            cur_fail++;
            if (cur_fail <= 10) $display("CUR FAIL img=%0d t=%0d k=%0d got=%h exp=%h", m_img, m_step, upd_count,
                                         dut.cur2[upd_count], trm[tidx(m_img, m_step, upd_count, 0)][32:0]);
          end
          if (dut.lif_mem_new !== trm[tidx(m_img, m_step, upd_count, 2)]) begin
            mem_out_fail++;
            if (mem_out_fail <= 10) $display("MEM_OUT FAIL img=%0d t=%0d k=%0d got=%h exp=%h", m_img, m_step, upd_count,
                                             dut.lif_mem_new, trm[tidx(m_img, m_step, upd_count, 2)]);
          end
          if (dut.lif_spike !== trm[tidx(m_img, m_step, upd_count, 3)][0]) begin
            spk_fail++;
            if (spk_fail <= 10) $display("SPIKE FAIL img=%0d t=%0d k=%0d got=%b exp=%b", m_img, m_step, upd_count,
                                         dut.lif_spike, trm[tidx(m_img, m_step, upd_count, 3)][0]);
          end
        end
        upd_count++;
      end
      if (ph == 1 && dut.state === 2'd1) begin       // S_CLEAR
        clr_wr_checks++;
        if (clr_count >= 10 || dut.kidx !== 4'(clr_count)) begin
          clr_wr_fail++;
          if (clr_wr_fail <= 10) $display("CLEAR ORDER FAIL k=%0d got %0d", clr_count, dut.kidx);
        end
        clr_count++;
      end
      if (step_done) begin
        sd_count++; stepdone_checks++;
        if (upd_count != 10) begin stepdone_fail++; $display("STEP_DONE with only %0d updates", upd_count); end
      end
      if (clear_done) begin
        cd_count++;
        if (clr_count != 10) begin clr_wr_fail++; $display("CLEAR_DONE with only %0d writes", clr_count); end
      end
    end
  end

  // ---------------- helpers ----------------
  task automatic do_reset();
    rst = 1'b1; e_start = 1'b0; clear_state = 1'b0; start_step = 1'b0; b_clear = 1'b0; b_start = 1'b0;
    hidden_spikes = '0;
    for (int k = 0; k < 10; k++) bcur[k] = '0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    @(posedge clk); #1;
  endtask

  function automatic bit all_mem_zero();
    for (int k = 0; k < 10; k++) if (dut.mem2[k] !== '0) return 1'b0;
    return 1'b1;
  endfunction

  task automatic do_clear(input bit spam);
    int cyc;
    mon_en = 1'b0; ph = 1; clr_count = 0; cd_count = 0;
    clear_state = 1'b1;
    @(posedge clk);
    mon_en = 1'b1;
    #1 clear_state = 1'b0;
    cyc = 0;
    while (!clear_done && cyc < 100) begin
      start_step  = spam && (cyc == 0 || cyc == 4 || cyc == 8);
      clear_state = spam && (cyc == 2 || cyc == 5 || cyc == 9);
      if (cyc >= 1) begin busy_checks++; if (busy !== 1'b1) busy_fail++; end
      @(posedge clk); #1; cyc++;
      if (spam && (cyc == 1 || cyc == 3)) ign_checks++;
    end
    start_step = 1'b0; clear_state = 1'b0;
    lat_checks++;
    if (cyc != EXP_CLR) begin lat_fail++; $display("CLEAR latency %0d != %0d", cyc, EXP_CLR); end
    if (clears_run == 0 || cyc < min_clr) min_clr = cyc;
    if (clears_run == 0 || cyc > max_clr) max_clr = cyc;
    if (busy !== 1'b0) begin busy_fail++; $display("busy high at clear_done"); end
    // at the first cycle clear_done is visible everything must already be zero
    clr_zero_checks += 11;
    for (int k = 0; k < 10; k++) if (dut.mem2[k] !== '0) begin clr_zero_fail++; $display("mem2[%0d] not zero at clear_done", k); end
    if (output_spikes !== '0) begin clr_zero_fail++; $display("output_spikes not zero at clear_done"); end
    repeat (3) @(posedge clk); #1;
    mon_en = 1'b0;
    if (clr_count != 10 || cd_count != 1) begin clr_wr_fail++; $display("clear writes=%0d pulses=%0d", clr_count, cd_count); end
    clears_run++;
  endtask

  task automatic run_engine(input int hidx);
    int cyc;
    hidden_spikes = hsm[hidx];
    e_start = 1'b1;
    @(posedge clk); #1 e_start = 1'b0;
    cyc = 0;
    while (!e_done && cyc < 5000) begin @(posedge clk); #1; cyc++; end
    eng_runs++;
    if (cyc != EXP_ENG) begin eng_fail++; $display("ENGINE latency %0d != %0d", cyc, EXP_ENG); end
    // engine done and sweep inputs stable (cur2 is held by the engine)
    @(posedge clk); #1;
  endtask

  task automatic run_step(input int img, input int t, input bit spam);
    int cyc, bad;
    mon_en = 1'b0; ph = 2; m_img = img; m_step = t; upd_count = 0; sd_count = 0;
    start_step = 1'b1;
    @(posedge clk);
    mon_en = 1'b1;
    #1 start_step = 1'b0;
    cyc = 0;
    while (!step_done && cyc < 100) begin
      start_step  = spam && (cyc == 0 || cyc == 4 || cyc == 9);
      clear_state = spam && (cyc == 2 || cyc == 6 || cyc == 9);
      if (cyc >= 1) begin busy_checks++; if (busy !== 1'b1) busy_fail++; end
      @(posedge clk); #1; cyc++;
      if (spam && (cyc == 1 || cyc == 3)) ign_checks++;
    end
    start_step = 1'b0; clear_state = 1'b0;
    lat_checks++;
    if (cyc != EXP_STEP) begin lat_fail++; $display("STEP latency %0d != %0d", cyc, EXP_STEP); end
    if (steps_run == 0 || cyc < min_step) min_step = cyc;
    if (steps_run == 0 || cyc > max_step) max_step = cyc;
    if (busy !== 1'b0) begin busy_fail++; $display("busy high at step_done"); end
    // at the first cycle step_done is visible all ten membranes and spikes are final
    bad = 0;
    for (int k = 0; k < 10; k++) begin
      if (dut.mem2[k] !== trm[tidx(img, t, k, 2)]) bad++;
      if (output_spikes[k] !== trm[tidx(img, t, k, 3)][0]) bad++;
    end
    if (bad != 0) begin mem_out_fail += bad; $display("STEP img=%0d t=%0d: %0d mismatches at step_done", img, t, bad); end
    repeat (3) @(posedge clk); #1;
    mon_en = 1'b0;
    wr_count_checks++;
    if (upd_count != 10 || sd_count != 1) begin
      wr_count_fail++; $display("STEP img=%0d t=%0d: updates=%0d step_done pulses=%0d", img, t, upd_count, sd_count);
    end
    steps_run++;
  endtask

  task automatic run_image(input int img, input bit spam);
    do_clear(spam);
    for (int t = 0; t < NSTEP; t++) begin
      run_engine(img * NSTEP + t);
      // the real hardware cur2 must equal Python before it is fed to the LIF
      run_step(img, t, spam && (t < 3));
    end
    images_run++;
  endtask

  // directed: preload mem2 and cur on DUT B, run one step, check
  task automatic run_directed(input int grp);
    int bad, cyc;
    for (int k = 0; k < 10; k++) begin
      dutb.mem2[k] = dirm[(grp * 10 + k) * 4 + 0];
      bcur[k]      = dirm[(grp * 10 + k) * 4 + 1][32:0];
    end
    b_start = 1'b1;
    @(posedge clk); #1 b_start = 1'b0;
    cyc = 0;
    while (!b_step_done && cyc < 100) begin @(posedge clk); #1; cyc++; end
    bad = 0;
    for (int k = 0; k < 10; k++) begin
      dir_checks++;
      if (dutb.mem2[k] !== dirm[(grp * 10 + k) * 4 + 2] || b_spikes[k] !== dirm[(grp * 10 + k) * 4 + 3][0]) begin
        bad++; dir_fail++;
        if (dir_fail <= 10)
          $display("DIRECTED FAIL grp=%0d k=%0d mem_in=%h cur=%h got=%h/%b exp=%h/%b", grp, k,
                   dirm[(grp * 10 + k) * 4 + 0], dirm[(grp * 10 + k) * 4 + 1], dutb.mem2[k], b_spikes[k],
                   dirm[(grp * 10 + k) * 4 + 2], dirm[(grp * 10 + k) * 4 + 3][0]);
      end
    end
    // clear on DUT B: all membranes and output spikes must be zero at clear_done
    if (b_spikes !== '0 && b_clear_checks < 6) begin
      b_clear = 1'b1; @(posedge clk); #1 b_clear = 1'b0;
      cyc = 0;
      while (!b_clear_done && cyc < 100) begin @(posedge clk); #1; cyc++; end
      b_clear_checks++;
      for (int k = 0; k < 10; k++) if (dutb.mem2[k] !== '0) b_clear_fail++;
      if (b_spikes !== '0) b_clear_fail++;
    end
    repeat (2) @(posedge clk); #1;
  endtask

  // reset in the middle of a step: controller idle, spikes zero, mem2 cleared (documented choice)
  task automatic reset_test();
    mon_en = 1'b0;
    run_engine(30);
    start_step = 1'b1;
    @(posedge clk); #1 start_step = 1'b0;
    repeat (4) @(posedge clk);
    #1 rst = 1'b1;
    @(posedge clk); #1;
    rst_checks += 5;
    if (busy !== 1'b0) rst_fail++;
    if (output_spikes !== '0) rst_fail++;
    if (step_done !== 1'b0 || clear_done !== 1'b0) rst_fail++;
    if (!all_mem_zero()) rst_fail++;
    rst = 1'b0;
    repeat (3) @(posedge clk); #1;
    if (busy !== 1'b0) rst_fail++;
  endtask

  task automatic read_text();
    int a, b, c, d;
    longint unsigned x0, x1, x2, x3;
    $readmemh("vectors/l2l_hs.txt", hsm);
    fd = $fopen("vectors/l2l_trace.txt", "r");
    for (int i = 0; i < NIMG * NSTEP * 10; i++) begin
      rc = $fscanf(fd, "%h %h %h %h\n", x0, x1, x2, x3);
      trm[i * 4 + 0] = x0[35:0]; trm[i * 4 + 1] = x1[35:0]; trm[i * 4 + 2] = x2[35:0]; trm[i * 4 + 3] = x3[35:0];
    end
    $fclose(fd);
    fd = $fopen("vectors/l2l_directed.txt", "r");
    for (int i = 0; i < NDIR; i++) begin
      rc = $fscanf(fd, "%h %h %h %h\n", x0, x1, x2, x3);
      dirm[i * 4 + 0] = x0[35:0]; dirm[i * 4 + 1] = x1[35:0]; dirm[i * 4 + 2] = x2[35:0]; dirm[i * 4 + 3] = x3[35:0];
    end
    $fclose(fd);
  endtask

  initial begin
    read_text();
    mon_en = 1'b0; ph = 0;
    do_reset();

    // images 0..3: MNIST 0, MNIST 1, all-zero, all-255
    for (int im = 0; im < NIMG; im++) run_image(im, im == 0);

    // reset mid-step, then repeat image 0 after clear (persistence from a clean state)
    reset_test();
    run_image(0, 1'b0);

    // directed integration cases
    for (int g = 0; g < NDIR / 10; g++) run_directed(g);

    fails = mem_in_fail + mem_out_fail + spk_fail + cur_fail + order_fail + wr_count_fail + stepdone_fail +
            clr_wr_fail + clr_zero_fail + lat_fail + busy_fail + rst_fail + dir_fail + eng_fail + b_clear_fail;

    $display("images run (incl. repeat after reset)   : %0d", images_run);
    $display("time steps                              : %0d", steps_run);
    $display("l2_current_engine runs                  : %0d", eng_runs);
    $display("LIF neuron updates                      : %0d", updates);
    $display("membrane input comparisons              : %0d", mem_in_cmp);
    $display("hw cur2 comparisons                     : %0d", cur_cmp);
    $display("membrane output comparisons             : %0d", mem_out_cmp);
    $display("output spike comparisons                : %0d", spk_cmp);
    $display("output-neuron order checks              : %0d", order_checks);
    $display("per-step write-count checks             : %0d", wr_count_checks);
    $display("step_done timing checks                 : %0d", stepdone_checks);
    $display("clear operations / zero checks          : %0d / %0d (+%0d on DUT B)", clears_run, clr_zero_checks, b_clear_checks);
    $display("clear latency measured min,max          : %0d, %0d (expected %0d)", min_clr, max_clr, EXP_CLR);
    $display("step latency measured min,max           : %0d, %0d (expected %0d)", min_step, max_step, EXP_STEP);
    $display("busy checks / ignored commands          : %0d / %0d", busy_checks, ign_checks);
    $display("reset checks                            : %0d", rst_checks);
    $display("directed cases                          : %0d", dir_checks);
    $display("failures: min=%0d mout=%0d spk=%0d cur=%0d order=%0d wrcnt=%0d sd=%0d clrwr=%0d clrz=%0d lat=%0d busy=%0d rst=%0d dir=%0d eng=%0d bclr=%0d",
             mem_in_fail, mem_out_fail, spk_fail, cur_fail, order_fail, wr_count_fail, stepdone_fail,
             clr_wr_fail, clr_zero_fail, lat_fail, busy_fail, rst_fail, dir_fail, eng_fail, b_clear_fail);
    $display("total failures: %0d", fails);
    if (fails == 0) $display("PASS tb_l2_lif_sweep");
    else $fatal(1, "FAIL tb_l2_lif_sweep");
    $finish;
  end
endmodule
