// Self-checking testbench for l1_current_sweep + cur1_ram (RTL milestone 6).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv ../rtl/pixel_buffer.sv \
//            ../rtl/l1_mac.sv ../rtl/l1_neuron_engine.sv ../rtl/cur1_ram.sv \
//            ../rtl/l1_current_sweep.sv tb_l1_current_sweep.sv
//   vsim -c work.tb_l1_current_sweep -do "run -all; quit -f"
// EARLY=1 samples cache read data before the capturing edge (must FAIL).
`timescale 1ns/1ps

module tb_l1_current_sweep #(
    parameter bit EARLY = 1'b0
);
  import snn_params_pkg::*;

  localparam int NIMG    = 6;
  localparam int EXP_LAT = 201473;   // 1 + 256*787

  logic clk = 1'b0;
  always #5 clk = ~clk;

  logic        rst;
  logic        pixel_we;
  logic [9:0]  pixel_waddr;
  logic [7:0]  pixel_wdata;
  logic        start;
  logic        busy, done;
  logic [7:0]  cur1_raddr;
  logic [L1_ACC_WIDTH-1:0] cur1_rdata;

  l1_current_sweep dut (.*);

  logic [7:0]  imgmem [0:NIMG*784-1];
  logic [35:0] expm   [0:NIMG*256-1];

  // ---------------- counters ----------------
  int sweeps, fails;
  int wr_total, wr_addr_checks, wr_data_checks, wr_fail;
  int eng_start_checks, eng_start_fail, eng_done_total, eng_done_fail;
  int rb_checks, rb_fail, rd_lat_checks, rd_lat_fail, stream_checks, stream_fail;
  int lat_fail, done_pulse_fail, busy_fail, ignore_fail, reset_fail;
  int min_lat, max_lat;

  // ---------------- monitor ----------------
  bit mon_en;
  int cur_img;
  int wr_count, start_count, done_count, sweep_done_cycles;

  always @(negedge clk) begin
    if (mon_en) begin
      if (dut.cur1_we) begin
        wr_total++;
        wr_addr_checks++; wr_data_checks++;
        if (wr_count >= 256 || dut.cur1_waddr !== 8'(wr_count)) begin
          wr_fail++;
          if (wr_fail <= 10) $display("WRITE ADDR FAIL count=%0d addr=%0d", wr_count, dut.cur1_waddr);
        end else if (dut.cur1_wdata !== expm[cur_img * 256 + wr_count][34:0]) begin
          wr_fail++;
          if (wr_fail <= 10)
            $display("WRITE DATA FAIL img=%0d j=%0d data=%h exp=%h", cur_img, wr_count,
                     dut.cur1_wdata, expm[cur_img * 256 + wr_count][34:0]);
        end
        wr_count++;
      end
      if (dut.u_eng.accept) begin
        eng_start_checks++;
        if (start_count >= 256 || dut.eng_neuron !== 8'(start_count)) begin
          eng_start_fail++;
          if (eng_start_fail <= 10) $display("ENGINE START SEQ FAIL n=%0d got neuron=%0d", start_count, dut.eng_neuron);
        end
        start_count++;
      end
      if (dut.eng_done) begin
        eng_done_total++;
        done_count++;
      end
      if (done) sweep_done_cycles++;
    end
  end

  // ---------------- helpers ----------------
  task automatic do_reset();
    rst = 1'b1; start = 1'b0; pixel_we = 1'b0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    @(posedge clk); #1;
  endtask

  task automatic load_image(input int img);
    for (int i = 0; i < 784; i++) begin
      pixel_we = 1'b1; pixel_waddr = 10'(i); pixel_wdata = imgmem[img * 784 + i];
      @(posedge clk); #1;
    end
    pixel_we = 1'b0;
  endtask

  // spam: extra start pulses while busy (neuron 0, middle, neuron 255)
  task automatic run_sweep(input int img, input bit spam, output int latency);
    int cyc;
    bit seen;
    mon_en = 1'b0;
    wr_count = 0; start_count = 0; done_count = 0; sweep_done_cycles = 0; cur_img = img;
    start = 1'b1;
    @(posedge clk);                        // accepted start edge
    mon_en = 1'b1;
    #1 start = 1'b0;
    cyc = 0; seen = 1'b0;
    while (!seen && cyc < 300000) begin
      start = spam && (cyc == 300 || cyc == 100000 || cyc == 787 * 255 + 400 || cyc == 201000);
      @(posedge clk); #1; cyc++;
      if (cyc > 1 && busy !== 1'b1 && !done) busy_fail++;
      if (done) seen = 1'b1;
    end
    start = 1'b0;
    latency = cyc;
    sweeps++;
    if (!seen) begin fails++; $display("SWEEP TIMEOUT img=%0d", img); end
    else begin
      if (cyc != EXP_LAT) begin lat_fail++; $display("LATENCY %0d != %0d", cyc, EXP_LAT); end
      if (busy !== 1'b0) begin busy_fail++; $display("busy high at sweep done"); end
    end
    repeat (4) @(posedge clk);
    #1;
    if (sweep_done_cycles != 1) begin done_pulse_fail++; $display("done high for %0d cycles", sweep_done_cycles); end
    if (wr_count != 256)    begin wr_fail++; $display("cache writes = %0d (img %0d)", wr_count, img); end
    if (start_count != 256) begin eng_start_fail++; $display("engine starts = %0d (img %0d)", start_count, img); end
    if (done_count != 256)  begin eng_done_fail++; $display("engine dones = %0d (img %0d)", done_count, img); end
    mon_en = 1'b0;
  endtask

  // cache readback: address before edge N -> data after edge N
  task automatic read_check(input int img, input int j);
    cur1_raddr = 8'(j);
    if (EARLY) #1;
    else begin @(posedge clk); #1; end
    rb_checks++;
    if (cur1_rdata !== expm[img * 256 + j][34:0]) begin
      rb_fail++;
      if (rb_fail <= 10)
        $display("READBACK FAIL img=%0d j=%0d data=%h exp=%h", img, j, cur1_rdata, expm[img * 256 + j][34:0]);
    end
  endtask

  task automatic verify_cache(input int img);
    int dj [0:6];
    dj = '{0, 1, 2, 127, 128, 254, 255};
    // directed addresses
    for (int k = 0; k < 7; k++) read_check(img, dj[k]);
    // all 256, one read each
    for (int j = 0; j < 256; j++) read_check(img, j);
    if (!EARLY) begin
      // latency: data keeps the previous address' value until the next edge
      cur1_raddr = 8'd10; @(posedge clk); #1;
      cur1_raddr = 8'd200; #3;
      rd_lat_checks++; if (cur1_rdata !== expm[img * 256 + 10][34:0]) rd_lat_fail++;
      @(posedge clk); #1;
      rd_lat_checks++; if (cur1_rdata !== expm[img * 256 + 200][34:0]) rd_lat_fail++;
      // back-to-back stream over all 256 entries
      cur1_raddr = 8'd0; @(posedge clk); #1;
      for (int j = 1; j < 256; j++) begin
        cur1_raddr = 8'(j); #3;
        stream_checks++; if (cur1_rdata !== expm[img * 256 + j - 1][34:0]) stream_fail++;
        @(posedge clk); #1;
        stream_checks++; if (cur1_rdata !== expm[img * 256 + j][34:0]) stream_fail++;
      end
    end
  endtask

  int k, lat, total_fail;
  int order [0:8];

  initial begin
    sweeps = 0; fails = 0; wr_total = 0; wr_addr_checks = 0; wr_data_checks = 0; wr_fail = 0;
    eng_start_checks = 0; eng_start_fail = 0; eng_done_total = 0; eng_done_fail = 0;
    rb_checks = 0; rb_fail = 0; rd_lat_checks = 0; rd_lat_fail = 0; stream_checks = 0; stream_fail = 0;
    lat_fail = 0; done_pulse_fail = 0; busy_fail = 0; ignore_fail = 0; reset_fail = 0;
    min_lat = 1 << 30; max_lat = 0; mon_en = 1'b0;
    cur_img = 0; wr_count = 0; start_count = 0; done_count = 0; sweep_done_cycles = 0;
    pixel_we = 1'b0; pixel_waddr = '0; pixel_wdata = '0; start = 1'b0; cur1_raddr = '0; rst = 1'b1;

    $readmemh("vectors/l1s_images.txt", imgmem);
    $readmemh("vectors/l1s_expected.txt", expm);
    do_reset();

    if (busy !== 1'b0 || done !== 1'b0) begin fails++; $display("busy/done not idle after reset"); end

    // images 0..3 (real), 4 (all-zero), 5 (all-255), then image 0 AGAIN: no stale data
    order = '{0, 1, 2, 3, 4, 5, 0, 2, 3};
    for (k = 0; k < 9; k++) begin
      load_image(order[k]);
      // sweep 7 (index 7): start pulses while busy; sweep 8 follows a mid-sweep reset below
      if (k == 8) begin
        // abort a sweep with a reset in flight, then sweep again
        mon_en = 1'b0;
        start = 1'b1; @(posedge clk); #1 start = 1'b0;
        repeat (5000) @(posedge clk);
        #1 rst = 1'b1; repeat (2) @(posedge clk); #1 rst = 1'b0;
        if (busy !== 1'b0 || done !== 1'b0) reset_fail++;
        repeat (2000) begin @(posedge clk); #1; if (busy !== 1'b0 || done !== 1'b0) reset_fail++; end
        load_image(order[k]);
      end
      run_sweep(order[k], (k == 7), lat);
      if (lat < min_lat) min_lat = lat;
      if (lat > max_lat) max_lat = lat;
      verify_cache(order[k]);
    end
    fails += reset_fail;

    total_fail = fails + wr_fail + eng_start_fail + eng_done_fail + rb_fail + rd_lat_fail +
                 stream_fail + lat_fail + done_pulse_fail + busy_fail;

    $display("----------------------------------------------");
    $display("sweeps (images)                 : %0d  (6 distinct images, 3 repeats, 1 start-spam, 1 after reset)", sweeps);
    $display("complete neuron calculations    : %0d", eng_done_total);
    $display("MAC products represented        : %0d", eng_done_total * 784);
    $display("cache writes                    : %0d  failures: %0d", wr_total, wr_fail);
    $display("write-address checks            : %0d", wr_addr_checks);
    $display("write-data checks               : %0d", wr_data_checks);
    $display("engine start-sequence checks    : %0d  failures: %0d", eng_start_checks, eng_start_fail);
    $display("engine done checks              : failures: %0d", eng_done_fail);
    $display("cache read-back checks          : %0d  failures: %0d", rb_checks, rb_fail);
    $display("cache read-latency checks       : %0d  failures: %0d", rd_lat_checks, rd_lat_fail);
    $display("cache back-to-back checks       : %0d  failures: %0d", stream_checks, stream_fail);
    $display("sweep latency (accepted start -> done): min %0d max %0d (expected %0d), failures %0d",
             min_lat, max_lat, EXP_LAT, lat_fail);
    $display("done-pulse failures %0d  busy failures %0d  reset failures %0d",
             done_pulse_fail, busy_fail, reset_fail);
    $display("total failures: %0d", total_fail);
    if (total_fail == 0) $display("PASS: tb_l1_current_sweep");
    else $fatal(1, "FAIL: tb_l1_current_sweep (%0d failures)", total_fail);
    $finish;
  end
endmodule
