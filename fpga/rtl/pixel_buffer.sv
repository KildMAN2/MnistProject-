// -----------------------------------------------------------------------------
// pixel_buffer : one MNIST image (784 x unsigned 8-bit), write port + synchronous
// read port (RTL milestone 5). Vendor-neutral inferred memory.
//
// Read: read_data <= mem[read_addr] on posedge clk -> address presented before
// edge N, data valid after edge N (1-cycle latency, same as sync_rom).
// Write: mem[write_addr] <= write_data when write_enable on posedge clk.
// Read-during-write to the SAME address is not used (the image is fully loaded
// before inference starts) and its result is not defined by this module.
// -----------------------------------------------------------------------------
module pixel_buffer #(
    parameter int WIDTH      = 8,
    parameter int DEPTH      = 784,
    parameter int ADDR_WIDTH = 10
) (
    input  logic                  clk,
    input  logic                  write_enable,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [WIDTH-1:0]      write_data,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [WIDTH-1:0]      read_data
);

  logic [WIDTH-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (write_enable) mem[write_addr] <= write_data;
    read_data <= mem[read_addr];
  end

endmodule
