// -----------------------------------------------------------------------------
// mem1_ram : persistent layer-1 membrane state, 256 x 38 bit (RTL milestone 7).
// Vendor-neutral inferred RAM, raw two's-complement bits.
//
// Write : mem[waddr] <= wdata when we, on posedge clk.
// Read  : rdata <= mem[raddr] on posedge clk (address before edge N, data after
//         edge N, 1-cycle latency).
// Read-during-write to the same address is not used and not defined here.
// There is NO reset/initialisation loop (keeps block-RAM inference possible):
// the contents are undefined at power-up and are zeroed explicitly by the
// clear_state operation of l1_lif_sweep before every new image.
// -----------------------------------------------------------------------------
module mem1_ram #(
    parameter int WIDTH      = 38,
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
