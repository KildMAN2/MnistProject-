// -----------------------------------------------------------------------------
// cur1_ram : cache of the 256 layer-1 currents of the loaded image
// (RTL milestone 6). Vendor-neutral inferred RAM, raw two's-complement bits.
//
// Write : mem[waddr] <= wdata when we, on posedge clk.
// Read  : rdata <= mem[raddr] on posedge clk -> address presented before edge N,
//         data valid after edge N (1-cycle latency, same as sync_rom).
// Read-during-write to the same address is not used and not defined here.
// The RAM is NOT cleared by reset: every sweep overwrites all 256 entries before
// anything reads them.
// -----------------------------------------------------------------------------
module cur1_ram #(
    parameter int WIDTH      = 35,
    parameter int DEPTH      = 256,
    parameter int ADDR_WIDTH = 8
) (
    input  logic                  clk,
    input  logic                  we,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [WIDTH-1:0]      wdata,
    input  logic [ADDR_WIDTH-1:0] raddr,
    output logic [WIDTH-1:0]      rdata
);

  logic [WIDTH-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end

endmodule
