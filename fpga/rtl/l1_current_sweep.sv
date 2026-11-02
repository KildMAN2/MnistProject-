// -----------------------------------------------------------------------------
// l1_current_sweep : computes cur1[j] for all 256 hidden neurons of the loaded
// image by reusing ONE l1_neuron_engine sequentially, and caches them in cur1_ram
// (RTL milestone 6).   cur1[j] = bias1[j] + sum_i pixel[i]*weight1[j][i]
//
// Interface
//   clk, rst                 clock, synchronous active-high reset
//   pixel_we/waddr/wdata     pixel load port (passed to the engine's pixel buffer)
//   start                    accepted only when busy = 0
//   busy                     high from the accepting edge until done
//   done                     ONE-clock pulse, registered: the cache is already
//                            completely written when it is seen
//   cur1_raddr / cur1_rdata  synchronous cache read (1-cycle latency)
//
// States: S_IDLE -> S_KICK -> S_WAIT -> (S_WAIT ...) -> S_IDLE
//   S_IDLE : start accepted: nidx <= 0, -> S_KICK
//   S_KICK : eng_start = 1 for neuron 0 (engine is IDLE, accepts at this edge) -> S_WAIT
//   S_WAIT : waits for the engine's done pulse. In the cycle where eng_done = 1
//            the engine is already back in IDLE and current_out is final:
//              cur1_we    = 1, cur1_waddr = nidx (OLD index), cur1_wdata = current_out
//              if nidx != 255: eng_start = 1 with neuron_index = nidx+1 in the same
//                              cycle (accepted at the same edge), nidx <= nidx+1
//              else           : state <= S_IDLE, done <= 1
//   All of this happens at one clock edge with nonblocking assignments, so the RAM
//   write uses the pre-increment nidx and the engine accepts the next neuron at the
//   very edge where the current result is stored.
//
// Schedule: engine latency 786 (accepted start edge -> done visible) + 1 cycle
// (the done cycle itself, during which the next start is accepted) = 787 clocks per
// neuron, 1 extra clock for S_KICK and the final registered done:
//   accepted sweep start edge -> done visible = 1 + 256*787 = 201,473 clocks.
// -----------------------------------------------------------------------------
module l1_current_sweep
  import snn_params_pkg::*;
(
    input  logic                           clk,
    input  logic                           rst,

    input  logic                           pixel_we,
    input  logic [9:0]                     pixel_waddr,
    input  logic [7:0]                     pixel_wdata,

    input  logic                           start,
    output logic                           busy,
    output logic                           done,

    input  logic [7:0]                     cur1_raddr,
    output logic [L1_ACC_WIDTH-1:0]        cur1_rdata
);

  typedef enum logic [1:0] {S_IDLE, S_KICK, S_WAIT} state_t;
  state_t state;

  logic [7:0] nidx;
  logic       last;
  assign last = (nidx == 8'(HIDDEN_SIZE - 1));
  assign busy = (state != S_IDLE);

  logic                           eng_start, eng_done, eng_busy;
  logic [7:0]                     eng_neuron;
  logic signed [L1_ACC_WIDTH-1:0] eng_current;

  logic                    cur1_we;
  logic [7:0]              cur1_waddr;
  logic [L1_ACC_WIDTH-1:0] cur1_wdata;

  assign cur1_we    = (state == S_WAIT) && eng_done;
  assign eng_start  = (state == S_KICK) || (cur1_we && !last);
  assign eng_neuron = (state == S_WAIT) ? (nidx + 8'd1) : nidx;
  assign cur1_waddr = nidx;
  assign cur1_wdata = eng_current;

  always_ff @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE;
      nidx  <= '0;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            nidx  <= '0;
            state <= S_KICK;
          end
        end
        S_KICK: state <= S_WAIT;
        S_WAIT: begin
          if (eng_done) begin
            if (last) begin
              state <= S_IDLE;
              done  <= 1'b1;
            end else begin
              nidx <= nidx + 8'd1;
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  l1_neuron_engine u_eng (
      .clk(clk), .rst(rst),
      .pixel_we(pixel_we), .pixel_waddr(pixel_waddr), .pixel_wdata(pixel_wdata),
      .start(eng_start), .neuron_index(eng_neuron),
      .busy(eng_busy), .done(eng_done), .current_out(eng_current));

  cur1_ram #(.WIDTH(L1_ACC_WIDTH), .DEPTH(HIDDEN_SIZE), .ADDR_WIDTH(8)) u_cur1 (
      .clk(clk), .we(cur1_we), .waddr(cur1_waddr), .wdata(cur1_wdata),
      .raddr(cur1_raddr), .rdata(cur1_rdata));

endmodule
