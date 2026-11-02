// Generic synchronous ROM (vendor-neutral, inferred).
//
// Timing: address presented before clock edge N -> data valid after edge N
// (1-cycle registered read). `data` keeps its value until the next edge, even if
// `addr` changes in between, so a consumer must delay its other signals
// (pixel index, valid, ...) by one cycle to stay aligned with the data.
//
// Contents are raw bits; consumers interpret them with $signed(data).
// Initialised from INIT_FILE with $readmemh (one hex word per line, address 0 first).
module sync_rom #(
    parameter int    WIDTH      = 16,
    parameter int    DEPTH      = 256,
    parameter int    ADDR_WIDTH = $clog2(DEPTH),
    parameter string INIT_FILE  = ""
) (
    input  logic                  clk,
    input  logic [ADDR_WIDTH-1:0] addr,
    output logic [WIDTH-1:0]      data
);

  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
  end

  always_ff @(posedge clk) begin
    data <= mem[addr];
  end

endmodule
