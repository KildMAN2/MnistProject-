// Self-checking testbench for decay_19_20 (MEM_WIDTH = 38).
// Golden vectors come from fpga/scripts/generate_decay_vectors.py, which uses
// the real Python reference _round_ratio_ties_to_even(m*19, 20).
// Run with the working directory = fpga/tb (see docs/TASK2_FPGA_EXPLANATION.md).
`timescale 1ns / 1ps

module tb_decay_19_20;

  localparam int MEM_WIDTH = 38;

  logic signed [MEM_WIDTH-1:0] membrane_in;
  logic signed [MEM_WIDTH-1:0] decayed_out;

  decay_19_20 #(.MEM_WIDTH(MEM_WIDTH)) dut (
      .membrane_in(membrane_in),
      .decayed_out(decayed_out)
  );

  integer total    = 0;
  integer failures = 0;
  integer directed = 0;

  task automatic check(input integer idx, input logic signed [MEM_WIDTH-1:0] m,
                       input logic signed [MEM_WIDTH-1:0] expected);
    begin
      membrane_in = m;
      #1;
      total = total + 1;
      if (decayed_out !== expected) begin
        failures = failures + 1;
        if (failures <= 20)
          $display("MISMATCH test %0d: in=%0d (0x%h) expected=%0d (0x%h) actual=%0d (0x%h)",
                   idx, m, m, expected, expected, decayed_out, decayed_out);
      end
    end
  endtask

  task automatic directed_case(input logic signed [MEM_WIDTH-1:0] m,
                               input logic signed [MEM_WIDTH-1:0] expected);
    begin
      directed = directed + 1;
      check(-directed, m, expected);
    end
  endtask

  integer fd, rc, idx;
  logic [MEM_WIDTH-1:0] in_bits, exp_bits;

  initial begin
    // ---- directed tests (expected values from the Python reference) ---------
    directed_case(38'sd0, 38'sd0);
    directed_case(38'sd20, 38'sd19);
    directed_case(-38'sd20, -38'sd19);
    directed_case(38'sd1, 38'sd1);        // 0.95  -> 1
    directed_case(-38'sd1, -38'sd1);
    directed_case(38'sd2, 38'sd2);        // 1.9   -> 2
    directed_case(38'sd3, 38'sd3);        // 2.85  -> 3
    directed_case(38'sd9, 38'sd9);        // 8.55  -> 9
    directed_case(38'sd11, 38'sd10);      // 10.45 -> 10
    // ties: 19*m/20 = N + 0.5 happens only for m = 20t + 10
    directed_case(38'sd10, 38'sd10);      //  9.5  -> 10 (even)
    directed_case(38'sd30, 38'sd28);      // 28.5  -> 28 (even)
    directed_case(38'sd50, 38'sd48);      // 47.5  -> 48 (even)
    directed_case(38'sd70, 38'sd66);      // 66.5  -> 66 (even)
    directed_case(-38'sd10, -38'sd10);    // -9.5  -> -10
    directed_case(-38'sd30, -38'sd28);    // -28.5 -> -28
    directed_case(-38'sd50, -38'sd48);    // -47.5 -> -48
    directed_case(-38'sd70, -38'sd66);    // -66.5 -> -66
    // extremes, including the most-negative two's-complement value
    directed_case(-38'sd137438953472, -38'sd130567005798);
    directed_case(38'sd137438953471, 38'sd130567005797);

    // ---- file-driven vectors -------------------------------------------------
    fd = $fopen("vectors/decay_19_20_vectors.txt", "r");
    if (fd == 0) $fatal(1, "cannot open vectors/decay_19_20_vectors.txt (run from fpga/tb)");
    idx = 0;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%h %h\n", in_bits, exp_bits);
      if (rc == 2) begin
        check(idx, in_bits, exp_bits);
        idx = idx + 1;
      end
    end
    $fclose(fd);

    $display("directed tests : %0d", directed);
    $display("file vectors   : %0d", idx);
    $display("total tests    : %0d", total);
    $display("failures       : %0d", failures);
    if (idx == 0) $fatal(1, "no vectors were read");
    if (failures == 0) $display("PASS: tb_decay_19_20");
    else $fatal(1, "FAIL: tb_decay_19_20 had %0d mismatches", failures);
    $finish;
  end

endmodule
