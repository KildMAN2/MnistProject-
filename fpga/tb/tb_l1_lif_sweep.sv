// Self-checking testbench for l1_lif_sweep + mem1_ram (RTL milestone 7).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv ../rtl/pixel_buffer.sv \
//            ../rtl/l1_mac.sv ../rtl/l1_neuron_engine.sv ../rtl/cur1_ram.sv \
//            ../rtl/l1_current_sweep.sv ../rtl/decay_19_20.sv ../rtl/lif_core.sv \
//            ../rtl/mem1_ram.sv ../rtl/l1_lif_sweep.sv tb_l1_lif_sweep.sv
//   vsim -c work.tb_l1_lif_sweep -do "run -all; quit -f"
//
// DUT A (dut)  : l1_lif_sweep fed by the REAL cur1 cache of l1_current_sweep (cs).
//                Flow per image: load pixels -> current sweep -> clear_state -> 25 x start_step.
// DUT B (dutb) : a second l1_lif_sweep fed by a testbench cur1 model; mem1 is preloaded
//                hierarchically to run directed (membrane, current) cases in one step.
// Check timing: results are examined #1 after a posedge (NBA updates visible);
// monitors sample at negedge.
`timescale 1ns/1ps

module tb_l1_lif_sweep;
  import snn_params_pkg::*;

  localparam int NIMG     = 4;
  localparam int NSTEP    = 25;
  localparam int EXP_CLR  = 256;
  localparam int EXP_STEP = 512;
  localparam int EXP_SWEEP = 201473;
  localparam int NDIR     = 1024;

  logic clk = 1'b0;
  always #5 clk = ~clk;
  logic rst;

  // ---------------- DUT A + current sweep ----------------
  logic        pixel_we;
  logic [9:0]  pixel_waddr;
  logic [7:0]  pixel_wdata;
  logic        cs_start, cs_busy, cs_done;
  logic [7:0]  cur1_raddr;
  logic [L1_ACC_WIDTH-1:0] cur1_rdata;

  l1_current_sweep cs (
      .clk(clk), .rst(rst), .pixel_we(pixel_we), .pixel_waddr(pixel_waddr),
      .pixel_wdata(pixel_wdata), .start(cs_start), .busy(cs_busy), .done(cs_done),
      .cur1_raddr(cur1_raddr), .cur1_rdata(cur1_rdata));

  logic clear_state, clear_done, start_step, busy, step_done;
  logic [HIDDEN_SIZE-1:0] hidden_spikes;

  l1_lif_sweep dut (
      .clk(clk), .rst(rst), .clear_state(clear_state), .clear_done(clear_done),
      .start_step(start_step), .busy(busy), .step_done(step_done),
      .cur1_raddr(cur1_raddr), .cur1_rdata(cur1_rdata), .hidden_spikes(hidden_spikes));

  // ---------------- DUT B + testbench cur1 model ----------------
  logic b_clear, b_clear_done, b_start, b_busy, b_step_done;
  logic [7:0] b_raddr;
  logic [L1_ACC_WIDTH-1:0] b_rdata;
  logic [HIDDEN_SIZE-1:0] b_spikes;
  logic [L1_ACC_WIDTH-1:0] bcur [0:255];
  always_ff @(posedge clk) b_rdata <= bcur[b_raddr];

  l1_lif_sweep dutb (
      .clk(clk), .rst(rst), .clear_state(b_clear), .clear_done(b_clear_done),
      .start_step(b_start), .busy(b_busy), .step_done(b_step_done),
      .cur1_raddr(b_raddr), .cur1_rdata(b_rdata), .hidden_spikes(b_spikes));

  // ---------------- vectors ----------------
  logic [7:0]  imgm [0:NIMG*784-1];
  logic [37:0] curm [0:NIMG*256-1];
  logic [37:0] trm  [0:NIMG*NSTEP*256*3-1];
  logic [37:0] dirm [0:NDIR*4-1];

  // ---------------- counters ----------------
  int updates, mem_cmp, spk_cmp, mem_fail, spk_fail;
  int order_checks, order_fail, wr_count_checks, wr_count_fail;
  int stepdone_checks, stepdone_fail, clr_wr_checks, clr_wr_fail, clr_zero_checks, clr_zero_fail;
  int lat_checks, lat_fail, busy_checks, busy_fail, ign_checks, ign_fail;
  int rst_checks, rst_fail, dir_checks, dir_fail, cur_checks, cur_fail, in_checks, in_fail;
  int steps_run, images_run, clears_run;
  int fails;

  // ---------------- monitor ----------------
  bit  mon_en;
  int  ph;                    // 1 = clear, 2 = step
  int  m_img, m_step;
  int  upd_count, clr_count;
  int  sd_count, cd_count;

  function automatic int tidx(input int img, input int t, input int j, input int k);
    return (((img * NSTEP) + t) * 256 + j) * 3 + k;
  endfunction

  always @(negedge clk) begin
    if (mon_en) begin
      if (ph == 2 && dut.mem_we && dut.state === 2'd3) begin
        order_checks++;
        if (upd_count >= 256 || dut.nidx !== 8'(upd_count) || dut.mem_waddr !== 8'(upd_count)) begin
          order_fail++;
          if (order_fail <= 10) $display("ORDER FAIL expected j=%0d nidx=%0d waddr=%0d", upd_count, dut.nidx, dut.mem_waddr);
        end else begin
          updates++; mem_cmp++; spk_cmp++; in_checks++; cur_checks++;
          if (dut.mem_rdata !== trm[tidx(m_img, m_step, upd_count, 0)]) begin
            in_fail++;
            if (in_fail <= 10) $display("MEM_IN FAIL img=%0d t=%0d j=%0d got=%h exp=%h", m_img, m_step, upd_count,
                                        dut.mem_rdata, trm[tidx(m_img, m_step, upd_count, 0)]);
          end
          if (dut.cur1_rdata !== curm[m_img * 256 + upd_count][34:0]) begin
            cur_fail++;
            if (cur_fail <= 10) $display("CUR_IN FAIL img=%0d t=%0d j=%0d got=%h exp=%h", m_img, m_step, upd_count,
                                         dut.cur1_rdata, curm[m_img * 256 + upd_count][34:0]);
          end
          if (dut.mem_wdata !== trm[tidx(m_img, m_step, upd_count, 1)][37:0]) begin
            mem_fail++;
            if (mem_fail <= 10) $display("MEM_OUT FAIL img=%0d t=%0d j=%0d got=%h exp=%h", m_img, m_step, upd_count,
                                         dut.mem_wdata, trm[tidx(m_img, m_step, upd_count, 1)]);
          end
          if (dut.lif_spike !== trm[tidx(m_img, m_step, upd_count, 2)][0]) begin
            spk_fail++;
            if (spk_fail <= 10) $display("SPIKE FAIL img=%0d t=%0d j=%0d got=%b exp=%b", m_img, m_step, upd_count,
                                         dut.lif_spike, trm[tidx(m_img, m_step, upd_count, 2)][0]);
          end
        end
        upd_count++;
      end
      if (ph == 1 && dut.mem_we && dut.state === 2'd1) begin
        clr_wr_checks++;
        if (clr_count >= 256 || dut.mem_waddr !== 8'(clr_count) || dut.mem_wdata !== '0) begin
          clr_wr_fail++;
          if (clr_wr_fail <= 10) $display("CLEAR WRITE FAIL k=%0d waddr=%0d wdata=%h", clr_count, dut.mem_waddr, dut.mem_wdata);
        end
        clr_count++;
      end
      if (step_done) begin
        sd_count++;
        stepdone_checks++;
        if (upd_count != 256) begin
          stepdone_fail++;
          $display("STEP_DONE with only %0d writes done", upd_count);
        end
      end
      if (clear_done) begin
        cd_count++;
        if (clr_count != 256) begin
          clr_wr_fail++;
          $display("CLEAR_DONE with only %0d zero writes", clr_count);
        end
      end
    end
  end

  // ---------------- helpers ----------------
  task automatic do_reset();
    rst = 1'b1; cs_start = 1'b0; pixel_we = 1'b0; clear_state = 1'b0; start_step = 1'b0;
    b_clear = 1'b0; b_start = 1'b0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    @(posedge clk); #1;
  endtask

  task automatic load_image(input int img);
    for (int i = 0; i < 784; i++) begin
      pixel_we = 1'b1; pixel_waddr = 10'(i); pixel_wdata = imgm[img * 784 + i];
      @(posedge clk); #1;
    end
    pixel_we = 1'b0;
  endtask

  task automatic current_sweep();
    int cyc;
    cs_start = 1'b1;
    @(posedge clk); #1 cs_start = 1'b0;
    cyc = 0;
    while (!cs_done && cyc < 300000) begin @(posedge clk); #1; cyc++; end
    lat_checks++;
    if (cyc != EXP_SWEEP) begin lat_fail++; $display("CURRENT SWEEP latency %0d != %0d", cyc, EXP_SWEEP); end
    repeat (3) @(posedge clk); #1;
  endtask

  function automatic bit all_mem_zero();
    for (int j = 0; j < 256; j++) if (dut.u_mem1.mem[j] !== '0) return 1'b0;
    return 1'b1;
  endfunction

  task automatic do_clear(input bit spam);
    int cyc;
    bit nz;
    nz = !all_mem_zero();
    mon_en = 1'b0; ph = 1; clr_count = 0; cd_count = 0;
    clear_state = 1'b1;
    @(posedge clk);                      // accepting edge E0
    mon_en = 1'b1;
    #1 clear_state = 1'b0;
    cyc = 0;
    while (!clear_done && cyc < 2000) begin
      start_step = spam && (cyc == 10 || cyc == 100);
      clear_state = spam && (cyc == 20 || cyc == 200);
      if (spam && cyc >= 1) begin busy_checks++; if (busy !== 1'b1) busy_fail++; end
      @(posedge clk); #1; cyc++;
      if (spam && (cyc == 10 || cyc == 20)) ign_checks++;
    end
    start_step = 1'b0; clear_state = 1'b0;
    lat_checks++;
    if (cyc != EXP_CLR) begin lat_fail++; $display("CLEAR latency %0d != %0d", cyc, EXP_CLR); end
    if (busy !== 1'b0) begin busy_fail++; $display("busy high at clear_done"); end
    repeat (4) @(posedge clk); #1;
    mon_en = 1'b0;
    clear_checks_post(nz);
    clears_run++;
  endtask

  task automatic clear_checks_post(input bit was_nonzero);
    int z;
    z = 0;
    for (int j = 0; j < 256; j++) begin
      clr_zero_checks++;
      if (dut.u_mem1.mem[j] !== '0) begin clr_zero_fail++; z++; end
    end
    if (z != 0) $display("CLEAR left %0d nonzero membranes", z);
    clr_zero_checks++;
    if (hidden_spikes !== '0) begin clr_zero_fail++; $display("hidden_spikes not cleared"); end
    if (clr_count != 256) begin clr_wr_fail++; $display("clear wrote %0d words", clr_count); end
    if (cd_count != 1) begin clr_wr_fail++; $display("clear_done pulses=%0d", cd_count); end
  endtask

  task automatic run_step(input int img, input int t, input bit spam);
    int cyc, bad;
    mon_en = 1'b0; ph = 2; m_img = img; m_step = t; upd_count = 0; sd_count = 0;
    start_step = 1'b1;
    @(posedge clk);                      // accepting edge E0
    mon_en = 1'b1;
    #1 start_step = 1'b0;
    cyc = 0;
    while (!step_done && cyc < 2000) begin
      start_step  = spam && (cyc == 5 || cyc == 100 || cyc == 300);
      clear_state = spam && (cyc == 7 || cyc == 400 || cyc == 511);
      if (cyc >= 1) begin busy_checks++; if (busy !== 1'b1) busy_fail++; end
      @(posedge clk); #1; cyc++;
      if (spam && (cyc == 5 || cyc == 7)) ign_checks++;
    end
    start_step = 1'b0; clear_state = 1'b0;
    lat_checks++;
    if (cyc != EXP_STEP) begin lat_fail++; $display("STEP latency %0d != %0d", cyc, EXP_STEP); end
    if (busy !== 1'b0) begin busy_fail++; $display("busy high at step_done"); end
    // results visible exactly when step_done is seen
    bad = 0;
    for (int j = 0; j < 256; j++) begin
      if (dut.u_mem1.mem[j] !== trm[tidx(img, t, j, 1)][37:0]) bad++;
      if (hidden_spikes[j] !== trm[tidx(img, t, j, 2)][0]) bad++;
    end
    if (bad != 0) begin
      mem_fail += bad;
      $display("STEP img=%0d t=%0d: %0d state mismatches at step_done", img, t, bad);
    end
    repeat (3) @(posedge clk); #1;
    mon_en = 1'b0;
    wr_count_checks++;
    if (upd_count != 256 || sd_count != 1) begin
      wr_count_fail++;
      $display("STEP img=%0d t=%0d: writes=%0d step_done pulses=%0d", img, t, upd_count, sd_count);
    end
    steps_run++;
  endtask

  task automatic run_image(input int img, input bit spam);
    load_image(img);
    current_sweep();
    do_clear(spam);
    for (int t = 0; t < NSTEP; t++) run_step(img, t, spam && (t < 3));
    images_run++;
  endtask

  // directed: preload membranes (hierarchical) and currents, run ONE step on DUT B
  task automatic run_directed(input int set);
    int bad;
    for (int j = 0; j < 256; j++) begin
      dutb.u_mem1.mem[j] = dirm[(set * 256 + j) * 4 + 0];
      bcur[j]            = dirm[(set * 256 + j) * 4 + 1][34:0];
    end
    b_start = 1'b1;
    @(posedge clk); #1 b_start = 1'b0;
    while (!b_step_done) begin @(posedge clk); #1; end
    bad = 0;
    for (int j = 0; j < 256; j++) begin
      dir_checks++;
      if (dutb.u_mem1.mem[j] !== dirm[(set * 256 + j) * 4 + 2] ||
          b_spikes[j] !== dirm[(set * 256 + j) * 4 + 3][0]) begin
        bad++; dir_fail++;
        if (dir_fail <= 10)
          $display("DIRECTED FAIL set=%0d j=%0d mem_in=%h cur=%h got=%h/%b exp=%h/%b", set, j,
                   dirm[(set * 256 + j) * 4 + 0], dirm[(set * 256 + j) * 4 + 1],
                   dutb.u_mem1.mem[j], b_spikes[j], dirm[(set * 256 + j) * 4 + 2],
                   dirm[(set * 256 + j) * 4 + 3][0]);
      end
    end
    repeat (2) @(posedge clk); #1;
  endtask

  // reset in the middle of a step: controller returns to idle, RAM is NOT cleared
  task automatic reset_test();
    logic [L1_MEM_WIDTH-1:0] before_last;
    mon_en = 1'b0;
    before_last = dut.u_mem1.mem[255];
    start_step = 1'b1;
    @(posedge clk); #1 start_step = 1'b0;
    repeat (100) @(posedge clk);
    #1 rst = 1'b1;
    @(posedge clk); #1;
    rst_checks += 6;
    if (busy !== 1'b0) rst_fail++;
    if (hidden_spikes !== '0) rst_fail++;
    if (step_done !== 1'b0 || clear_done !== 1'b0) rst_fail++;
    rst = 1'b0;
    repeat (3) @(posedge clk); #1;
    if (busy !== 1'b0) rst_fail++;
    // neuron 255 was never reached: RAM must still hold the old value (reset does not clear it)
    if (dut.u_mem1.mem[255] !== before_last) rst_fail++;
    if (all_mem_zero()) rst_fail++;
  endtask

  initial begin
    $readmemh("vectors/l1l_images.txt", imgm);
    $readmemh("vectors/l1l_cur1.txt", curm);
    $readmemh("vectors/l1l_trace.txt", trm);
    $readmemh("vectors/l1l_directed.txt", dirm);
    mon_en = 1'b0; ph = 0;
    do_reset();

    // images 0..3: 0 = MNIST test 0, 1 = MNIST test 1, 2 = all-zero, 3 = all-255
    for (int im = 0; im < NIMG; im++) run_image(im, im == 0);

    // reset mid-step, then fresh clear and a full image again (image 0)
    reset_test();
    run_image(0, 1'b0);

    // directed integration cases
    for (int s = 0; s < NDIR / 256; s++) run_directed(s);

    fails = mem_fail + spk_fail + order_fail + wr_count_fail + stepdone_fail + clr_wr_fail +
            clr_zero_fail + lat_fail + busy_fail + ign_fail + rst_fail + dir_fail + cur_fail + in_fail;

    $display("images run (incl. repeat after reset)    : %0d", images_run);
    $display("LIF time steps run                       : %0d", steps_run);
    $display("neuron LIF updates checked               : %0d", updates);
    $display("membrane comparisons (write-data)        : %0d", mem_cmp);
    $display("spike comparisons                        : %0d", spk_cmp);
    $display("membrane_in / current_in checks          : %0d / %0d", in_checks, cur_checks);
    $display("neuron-order/waddr checks                : %0d", order_checks);
    $display("per-step write-count checks              : %0d", wr_count_checks);
    $display("step_done timing checks                  : %0d", stepdone_checks);
    $display("clears run / zero-write checks / zero checks: %0d / %0d / %0d", clears_run, clr_wr_checks, clr_zero_checks);
    $display("latency checks                           : %0d", lat_checks);
    $display("busy checks / ignored-command checks     : %0d / %0d", busy_checks, ign_checks);
    $display("reset checks                             : %0d", rst_checks);
    $display("directed cases                           : %0d", dir_checks);
    $display("failures: mem=%0d spk=%0d order=%0d wrcnt=%0d stepdone=%0d clrwr=%0d clrzero=%0d lat=%0d busy=%0d rst=%0d dir=%0d cur=%0d in=%0d",
             mem_fail, spk_fail, order_fail, wr_count_fail, stepdone_fail, clr_wr_fail, clr_zero_fail,
             lat_fail, busy_fail, rst_fail, dir_fail, cur_fail, in_fail);
    $display("total failures: %0d", fails);
    if (fails == 0) $display("PASS tb_l1_lif_sweep");
    else $fatal(1, "FAIL tb_l1_lif_sweep");
    $finish;
  end
endmodule
