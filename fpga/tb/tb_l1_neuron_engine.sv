// Self-checking testbench for pixel_buffer and l1_neuron_engine (RTL milestone 5).
// Run from fpga/tb:
//   vlog -sv ../rtl/snn_params_pkg.sv ../rtl/sync_rom.sv ../rtl/pixel_buffer.sv \
//            ../rtl/l1_mac.sv ../rtl/l1_neuron_engine.sv tb_l1_neuron_engine.sv
//   vsim -c work.tb_l1_neuron_engine -do "run -all; quit -f"
//
// Stimulus is applied 1 ns after a rising edge. Cycle-level monitors sample at the
// falling edge (mid-cycle), i.e. exactly the values the next rising edge will use.
`timescale 1ns/1ps

module tb_l1_neuron_engine;
  import snn_params_pkg::*;

  localparam int NIMG    = 10;
  localparam int MAXT    = 1024;
  localparam int EXP_LAT = 786;   // accepted-start edge -> edge after which done = 1

  logic clk = 1'b0;
  always #5 clk = ~clk;

  // ---------------- DUT ----------------
  logic        rst;
  logic        pixel_we;
  logic [9:0]  pixel_waddr;
  logic [7:0]  pixel_wdata;
  logic        start;
  logic [7:0]  neuron_index;
  logic        busy, done;
  logic signed [L1_ACC_WIDTH-1:0] current_out;

  l1_neuron_engine dut (.*);

  // ---------------- standalone pixel buffer under test ----------------
  logic       pb_we;
  logic [9:0] pb_waddr, pb_raddr;
  logic [7:0] pb_wdata, pb_rdata;
  pixel_buffer #(.WIDTH(8), .DEPTH(784), .ADDR_WIDTH(10)) u_pb_test (
      .clk(clk), .write_enable(pb_we), .write_addr(pb_waddr), .write_data(pb_wdata),
      .read_addr(pb_raddr), .read_data(pb_rdata));

  // ---------------- reference data ----------------
  logic [7:0]  imgmem [0:NIMG*784-1];
  logic [15:0] wref   [0:L1_WEIGHT_DEPTH-1];
  logic [31:0] bref   [0:L1_BIAS_DEPTH-1];
  int          t_img [0:MAXT-1];
  int          t_n   [0:MAXT-1];
  logic [35:0] t_exp [0:MAXT-1];
  int          nt;

  // ---------------- counters ----------------
  int pb_reads, pb_fail;
  int tests_run, fails, oracle_fail;
  int total_products, align_checks, align_fail;
  int bias_checks, extra_valid, multi_done, lat_fail, prod_cnt_fail, done_cnt_fail;
  int busy_ignore_tests, busy_ignore_fail, reset_tests, reset_fail;
  int min_lat, max_lat;

  // ---------------- monitor state ----------------
  bit          mon_en;
  int          cur_neuron;
  int          prod_idx;
  int          done_cnt;
  bit          prev_done;
  logic [9:0]  prev_pix_addr;
  logic [L1_WEIGHT_ADDR_WIDTH-1:0] prev_w_addr;
  int img_base;
  int          valid_total;

  always @(negedge clk) begin
    if (mon_en) begin
      if (dut.mac_load_bias) begin
        bias_checks++;
        if (dut.bias_data !== bref[cur_neuron]) begin
          align_fail++;
          $display("BIAS ALIGN FAIL neuron=%0d bias_data=%h exp=%h", cur_neuron, dut.bias_data, bref[cur_neuron]);
        end
      end
      if (dut.mac_valid) begin
        valid_total++;
        if (!busy) begin
          extra_valid++;
          $display("MAC valid while idle at %0t", $time);
        end else if (prod_idx >= 784) begin
          extra_valid++;
          $display("MAC valid beyond product 783 at %0t", $time);
        end else begin
          align_checks++;
          // addresses captured by the ROMs at the previous edge
          if (prev_pix_addr !== 10'(prod_idx) ||
              prev_w_addr   !== (L1_WEIGHT_ADDR_WIDTH)'(cur_neuron * 784 + prod_idx) ||
              dut.pix_data  !== imgmem[img_base + prod_idx] ||
              dut.w_data    !== wref[cur_neuron * 784 + prod_idx]) begin
            align_fail++;
            if (align_fail <= 10)
              $display("ALIGN FAIL idx=%0d neuron=%0d pix_addr=%0d w_addr=%0d pix=%h(exp %h) w=%h(exp %h)",
                       prod_idx, cur_neuron, prev_pix_addr, prev_w_addr, dut.pix_data,
                       imgmem[img_base + prod_idx], dut.w_data, wref[cur_neuron * 784 + prod_idx]);
          end
          prod_idx++;
        end
      end
      if (done) begin
        done_cnt++;
        if (prev_done) multi_done++;
      end
      prev_done = done;
    end
    prev_pix_addr = dut.pix_raddr;
    prev_w_addr   = dut.w_raddr;
  end

  // ---------------- helpers ----------------
  task automatic do_reset();
    rst = 1'b1; start = 1'b0; pixel_we = 1'b0;
    repeat (3) @(posedge clk);
    #1 rst = 1'b0;
    @(posedge clk); #1;
  endtask

  task automatic load_image(input int img);
    int i;
    for (i = 0; i < 784; i++) begin
      pixel_we = 1'b1; pixel_waddr = 10'(i); pixel_wdata = imgmem[img * 784 + i];
      @(posedge clk); #1;
    end
    pixel_we = 1'b0;
  endtask

  // returns the measured accepted-start -> done latency (clocks)
  task automatic run_test(input int img, input int neuron, input logic [35:0] exp_raw,
                          input int busy_start_at, output int latency);
    logic signed [34:0] exp_s;
    logic signed [34:0] got;
    logic signed [63:0] sum;
    int cyc, k;
    bit found;
    exp_s = exp_raw[34:0];
    // independent TB-side oracle from the reference arrays
    sum = $signed(bref[neuron]);
    for (k = 0; k < 784; k++) sum += 64'(imgmem[img * 784 + k]) * 64'($signed(wref[neuron * 784 + k]));
    if (sum !== 64'(exp_s)) begin
      oracle_fail++;
      $display("ORACLE MISMATCH img=%0d n=%0d tb=%0d file=%0d", img, neuron, sum, exp_s);
    end

    mon_en = 1'b0;
    prod_idx = 0; done_cnt = 0; prev_done = 1'b0;
    cur_neuron = neuron; img_base = img * 784;
    neuron_index = 8'(neuron); start = 1'b1;
    @(posedge clk);                       // E0: start accepted here
    mon_en = 1'b1;
    #1 start = 1'b0;
    cyc = 0; found = 1'b0;
    while (!found && cyc < 2000) begin
      if (cyc == busy_start_at) begin start = 1'b1; neuron_index = 8'(neuron) ^ 8'h55; end
      else begin start = 1'b0; end
      @(posedge clk); #1; cyc++;
      if (done) found = 1'b1;
    end
    start = 1'b0; neuron_index = 8'(neuron);
    got = current_out;
    latency = cyc;
    tests_run++;
    if (!found) begin
      fails++; $display("TIMEOUT img=%0d neuron=%0d", img, neuron);
    end else begin
      if (cyc != EXP_LAT) begin lat_fail++; fails++; $display("LATENCY %0d != %0d (n=%0d)", cyc, EXP_LAT, neuron); end
      if (got !== exp_s) begin
        fails++;
        $display("CURRENT MISMATCH img=%0d neuron=%0d expected=%0d (%h) actual=%0d (%h)",
                 img, neuron, exp_s, exp_s, got, got);
      end
      if (busy !== 1'b0) begin fails++; $display("busy still high at done"); end
    end
    repeat (3) @(posedge clk);
    #1;
    if (done !== 1'b0) begin multi_done++; end
    // monitor-derived checks (done pulses, product count)
    if (done_cnt != 1) begin done_cnt_fail++; fails++; $display("done pulses = %0d (n=%0d)", done_cnt, neuron); end
    if (prod_idx != 784) begin prod_cnt_fail++; fails++; $display("products = %0d (n=%0d)", prod_idx, neuron); end
    total_products += prod_idx;
    if (current_out !== exp_s) begin fails++; $display("current_out not stable after done (n=%0d)", neuron); end
    if (busy_start_at >= 0) begin
      busy_ignore_tests++;
      if (got !== exp_s || cyc != EXP_LAT) busy_ignore_fail++;
    end
    mon_en = 1'b0;
  endtask

  // ---------------- pixel buffer test ----------------
  task automatic pb_check(input int addr, input logic [7:0] exp, input string what);
    pb_raddr = 10'(addr);
    @(posedge clk); #1;
    pb_reads++;
    if (pb_rdata !== exp) begin
      pb_fail++;
      $display("PIXEL BUFFER MISMATCH %s addr=%0d expected=%h actual=%h", what, addr, exp, pb_rdata);
    end
  endtask

  task automatic pixel_buffer_test();
    int i;
    // directed first/last address
    pb_we = 1'b1; pb_waddr = 10'd0;   pb_wdata = 8'hA5; @(posedge clk); #1;
    pb_waddr = 10'd783;               pb_wdata = 8'h5C; @(posedge clk); #1;
    pb_waddr = 10'd1;                 pb_wdata = 8'hFF; @(posedge clk); #1;
    pb_waddr = 10'd782;               pb_wdata = 8'h00; @(posedge clk); #1;
    pb_we = 1'b0;
    pb_check(0, 8'hA5, "addr0");
    pb_check(783, 8'h5C, "addr783");
    pb_check(1, 8'hFF, "addr1");
    pb_check(782, 8'h00, "addr782");
    // latency: data keeps the old address' value until the next edge
    pb_raddr = 10'd0; @(posedge clk); #1;
    pb_raddr = 10'd783; #3;
    pb_reads++; if (pb_rdata !== 8'hA5) begin pb_fail++; $display("PIXEL BUFFER latency fail (mid-cycle)"); end
    @(posedge clk); #1;
    pb_reads++; if (pb_rdata !== 8'h5C) begin pb_fail++; $display("PIXEL BUFFER latency fail (after edge)"); end
    // full real images, written then read back byte by byte (two images)
    for (int im = 0; im < 2; im++) begin
      for (i = 0; i < 784; i++) begin
        pb_we = 1'b1; pb_waddr = 10'(i); pb_wdata = imgmem[im * 784 + i]; @(posedge clk); #1;
      end
      pb_we = 1'b0;
      for (i = 0; i < 784; i++) pb_check(i, imgmem[im * 784 + i], "image");
    end
  endtask

  // ---------------- main ----------------
  int fd, rc, k, lat, a, b, c, total_fail;
  logic [35:0] eh;
  int cur_img;

  function automatic logic [35:0] find_exp(input int img, input int n);
    for (int q = 0; q < nt; q++) if (t_img[q] == img && t_n[q] == n) return t_exp[q];
    $fatal(1, "no vector for img=%0d neuron=%0d", img, n);
    return '0;
  endfunction

  initial begin
    pb_reads = 0; pb_fail = 0; tests_run = 0; fails = 0; oracle_fail = 0;
    total_products = 0; align_checks = 0; align_fail = 0; bias_checks = 0;
    extra_valid = 0; multi_done = 0; lat_fail = 0; prod_cnt_fail = 0; done_cnt_fail = 0;
    busy_ignore_tests = 0; busy_ignore_fail = 0; reset_tests = 0; reset_fail = 0;
    min_lat = 1 << 30; max_lat = 0; valid_total = 0;
    mon_en = 1'b0; prev_done = 1'b0; prod_idx = 0; done_cnt = 0; cur_neuron = 0; img_base = 0;
    pb_we = 1'b0; pb_waddr = '0; pb_raddr = '0; pb_wdata = '0;
    pixel_we = 1'b0; pixel_waddr = '0; pixel_wdata = '0; start = 1'b0; neuron_index = '0; rst = 1'b1;

    $readmemh("vectors/l1e_images.txt", imgmem);
    $readmemh(L1_WEIGHT_FILE, wref);
    $readmemh(L1_BIAS_FILE, bref);
    fd = $fopen("vectors/l1e_tests.txt", "r");
    if (fd == 0) $fatal(1, "cannot open vectors/l1e_tests.txt");
    nt = 0;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%d %d %h\n", a, b, eh);
      if (rc == 3) begin t_img[nt] = a; t_n[nt] = b; t_exp[nt] = eh; nt++; end
    end
    $fclose(fd);
    $display("loaded %0d engine tests", nt);

    do_reset();

    // ---- pixel buffer ----
    pixel_buffer_test();

    // ---- idle behaviour after reset ----
    if (busy !== 1'b0 || done !== 1'b0) begin fails++; $display("busy/done not idle after reset"); end
    repeat (5) @(posedge clk);
    #1;
    if (busy !== 1'b0 || done !== 1'b0) begin fails++; $display("busy/done changed while idle"); end

    // ---- all engine tests ----
    cur_img = -1;
    for (k = 0; k < nt; k++) begin
      if (t_img[k] != cur_img) begin load_image(t_img[k]); cur_img = t_img[k]; end
      run_test(t_img[k], t_n[k], t_exp[k], -1, lat);
      if (lat < min_lat) min_lat = lat;
      if (lat > max_lat) max_lat = lat;
    end

    // ---- start while busy must be ignored (various phases of a run) ----
    load_image(2); cur_img = 2;
    for (k = 0; k < 6; k++) begin
      case (k)
        0: begin c = 0;   b = 0;   end   // BIAS state
        1: begin c = 1;   b = 1;   end   // LOAD state
        2: begin c = 2;   b = 2;   end   // first RUN cycle
        3: begin c = 100; b = 127; end
        4: begin c = 700; b = 128; end
        default: begin c = 785; b = 255; end   // DRAIN state
      endcase
      run_test(2, b, find_exp(2, b), c, lat);
    end

    // ---- reset in the middle of a computation, then a normal run ----
    reset_tests++;
    mon_en = 1'b0;
    cur_neuron = 127; img_base = 2 * 784;
    neuron_index = 8'd127; start = 1'b1;
    @(posedge clk); #1 start = 1'b0;
    repeat (300) @(posedge clk);
    #1 rst = 1'b1;
    repeat (2) @(posedge clk);
    #1 rst = 1'b0;
    if (busy !== 1'b0 || done !== 1'b0) reset_fail++;
    repeat (900) begin
      @(posedge clk); #1;
      if (done !== 1'b0 || busy !== 1'b0) reset_fail++;
    end
    run_test(2, 127, find_exp(2, 127), -1, lat);
    if (reset_fail != 0) begin fails += reset_fail; $display("reset-in-flight test failed"); end

    // ---- final report ----
    total_fail = fails + pb_fail + oracle_fail + align_fail + extra_valid + multi_done;
    $display("----------------------------------------------");
    $display("pixel-buffer reads checked      : %0d  mismatches: %0d", pb_reads, pb_fail);
    $display("neuron-current tests (total)    : %0d  (file tests %0d, busy-ignore %0d, +1 after reset)",
             tests_run, nt, busy_ignore_tests);
    $display("complete 784-product runs       : %0d", tests_run);
    $display("MAC products verified           : %0d", total_products);
    $display("MAC valid operations observed   : %0d  (extra/illegal: %0d)", valid_total, extra_valid);
    $display("address/data alignment checks   : %0d  failures: %0d", align_checks, align_fail);
    $display("bias-load alignment checks      : %0d", bias_checks);
    $display("accepted-start->done latency    : min %0d max %0d (expected %0d), latency failures %0d",
             min_lat, max_lat, EXP_LAT, lat_fail);
    $display("multi-cycle done pulses         : %0d  done-count failures: %0d", multi_done, done_cnt_fail);
    $display("product-count failures          : %0d", prod_cnt_fail);
    $display("TB oracle (arrays) mismatches   : %0d", oracle_fail);
    $display("busy-ignore failures            : %0d   reset failures: %0d", busy_ignore_fail, reset_fail);
    $display("total failures: %0d", total_fail);
    if (total_fail == 0) $display("PASS: tb_l1_neuron_engine");
    else $fatal(1, "FAIL: tb_l1_neuron_engine (%0d failures)", total_fail);
    $finish;
  end
endmodule
