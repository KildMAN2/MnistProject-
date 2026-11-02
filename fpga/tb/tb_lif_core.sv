// Self-checking testbench for lif_core.
// Golden vectors: fpga/scripts/generate_lif_vectors.py (oracle = Int16SNN._lif_step).
// Run with the working directory = fpga/tb.
`timescale 1ns / 1ps

// One layer configuration, driven from a vector file.
module tb_lif_layer #(
    parameter int     MEM_WIDTH = 38,
    parameter int     CUR_WIDTH = 35,
    parameter longint THRESHOLD = 37718198,
    parameter string  VEC_FILE  = "vectors/lif_l1_vectors.txt",
    parameter string  NAME      = "L1"
) (
    output logic        done,
    output integer      tests,
    output integer      failures
);

  logic signed [MEM_WIDTH-1:0] membrane_in, membrane_out;
  logic signed [CUR_WIDTH-1:0] current_in;
  logic                        spike_out;

  lif_core #(.MEM_WIDTH(MEM_WIDTH), .CUR_WIDTH(CUR_WIDTH), .THRESHOLD(THRESHOLD)) dut (
      .membrane_in(membrane_in), .current_in(current_in),
      .membrane_out(membrane_out), .spike_out(spike_out)
  );

  integer fd, rc;
  logic [MEM_WIDTH-1:0] m_bits, e_bits;
  logic [CUR_WIDTH-1:0] c_bits;
  logic                 e_spike;

  initial begin
    done = 0; tests = 0; failures = 0;
    #1;
    fd = $fopen(VEC_FILE, "r");
    if (fd == 0) $fatal(1, "%s: cannot open %s (run from fpga/tb)", NAME, VEC_FILE);
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%h %h %h %h\n", m_bits, c_bits, e_bits, e_spike);
      if (rc == 4) begin
        membrane_in = m_bits;
        current_in  = c_bits;
        #1;
        if (membrane_out !== $signed(e_bits) || spike_out !== e_spike) begin
          failures = failures + 1;
          if (failures <= 20)
            $display({"%s MISMATCH #%0d: thr=%0d\n  mem_in=%0d (0x%h) cur_in=%0d (0x%h)\n",
                      "  expected mem=%0d (0x%h) spike=%0b\n  actual   mem=%0d (0x%h) spike=%0b\n",
                      "  dut: decayed=%0d prev_reset=%0b base=%0d pre_spike=%0d"},
                     NAME, tests, THRESHOLD, membrane_in, m_bits, current_in, c_bits,
                     $signed(e_bits), e_bits, e_spike, membrane_out, membrane_out, spike_out,
                     dut.decayed, dut.prev_reset, dut.base, dut.pre_spike);
        end
        tests = tests + 1;
      end
    end
    $fclose(fd);
    if (tests == 0) $fatal(1, "%s: no vectors read", NAME);
    done = 1;
  end

endmodule

module tb_lif_core;

  logic   d1, d2;
  integer t1, f1, t2, f2;

  tb_lif_layer #(.MEM_WIDTH(38), .CUR_WIDTH(35), .THRESHOLD(37718198),
                 .VEC_FILE("vectors/lif_l1_vectors.txt"), .NAME("Layer1"))
      l1 (.done(d1), .tests(t1), .failures(f1));
  tb_lif_layer #(.MEM_WIDTH(36), .CUR_WIDTH(33), .THRESHOLD(164691),
                 .VEC_FILE("vectors/lif_l2_vectors.txt"), .NAME("Layer2"))
      l2 (.done(d2), .tests(t2), .failures(f2));

  // ---- regression for the earlier architecture bug (threshold = 100) ----------
  // membrane=120, decayed=114, current=50 -> prev_reset=1, base=164,
  // pre_spike=64, spike=0, membrane_new=164. The rejected form
  // "spike = (decayed+current) > thr" would give spike=1.
  logic signed [15:0] r_mem, r_mem_out;
  logic signed [11:0] r_cur;
  logic               r_spike;
  lif_core #(.MEM_WIDTH(16), .CUR_WIDTH(12), .THRESHOLD(100))
      reg_dut (.membrane_in(r_mem), .current_in(r_cur),
               .membrane_out(r_mem_out), .spike_out(r_spike));

  integer reg_fail = 0;

  task automatic expect_lif(input logic signed [15:0] m, input logic signed [11:0] c,
                            input logic signed [15:0] e_mem, input logic e_spk,
                            input logic e_prev, input logic signed [15:0] e_base,
                            input logic signed [15:0] e_pre);
    begin
      r_mem = m; r_cur = c; #1;
      if (r_mem_out !== e_mem || r_spike !== e_spk || reg_dut.prev_reset !== e_prev ||
          reg_dut.base !== e_base || reg_dut.pre_spike !== e_pre) begin
        reg_fail = reg_fail + 1;
        $display("REGRESSION MISMATCH m=%0d c=%0d: mem=%0d(exp %0d) spike=%0b(exp %0b) prev=%0b base=%0d pre=%0d",
                 m, c, r_mem_out, e_mem, r_spike, e_spk, reg_dut.prev_reset, reg_dut.base,
                 reg_dut.pre_spike);
      end
    end
  endtask

  initial begin
    #0.5;
    expect_lif(120, 50, 164, 0, 1, 164, 64);   // the regression case
    // strict '>' on pre_spike with prev_reset = 0: 99 -> 100 -> 101 vs thr 100
    expect_lif(0, 99,  99, 0, 0, 99, 99);
    expect_lif(0, 100, 100, 0, 0, 100, 100);
    expect_lif(0, 101, 1, 1, 0, 101, 101);
    // strict '>' on the stored membrane: 99, 100, 101
    expect_lif(100, 0, 95, 0, 0, 95, 95);      // 100 > 100 is false -> no prev_reset
    expect_lif(101, 0, 96, 0, 1, 96, -4);      // decay(101)=96, prev_reset=1
    // pre_spike = thr-1 / thr / thr+1 with prev_reset = 1 (m=120, decay=114)
    expect_lif(120, 85, 199, 0, 1, 199, 99);   // 99  -> no spike
    expect_lif(120, 86, 200, 0, 1, 200, 100);  // 100 -> no spike
    expect_lif(120, 87, 101, 1, 1, 201, 101);  // 101 -> spike, mem = 201-100
    // negative membrane and negative current
    expect_lif(-20, -5, -24, 0, 0, -24, -24);
  end

  initial begin
    wait (d1 && d2);
    #2;
    $display("Layer 1 (MEM=38 CUR=35 THR=37718198): tests = %0d  failures = %0d", t1, f1);
    $display("Layer 2 (MEM=36 CUR=33 THR=164691)  : tests = %0d  failures = %0d", t2, f2);
    $display("Regression (THR=100)                : failures = %0d", reg_fail);
    if (f1 == 0 && f2 == 0 && reg_fail == 0) $display("PASS: tb_lif_core");
    else $fatal(1, "FAIL: tb_lif_core");
    $finish;
  end

endmodule
