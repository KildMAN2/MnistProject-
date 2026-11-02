// -----------------------------------------------------------------------------
// l1_lif_sweep : ONE complete layer-1 LIF time step for all 256 hidden neurons,
// with persistent membrane state (RTL milestone 7).
//
// For every neuron j = 0..255 (strictly in this order):
//   membrane_old = mem1[j]   (stored BEFORE this update)
//   current      = cur1[j]   (read from the cur1 cache owned by someone else)
//   {membrane_new, spike} = lif_core(membrane_old, current)     (verified block)
//   mem1[j] <= membrane_new ; hidden_spikes[j] <= spike
//
// Interface
//   clk, rst            clock, synchronous active-high reset (controller only)
//   clear_state         pulse: write mem1[0..255] = 0 sequentially (accepted when idle)
//   clear_done          ONE-clock registered pulse after the last zero write
//   start_step          pulse: run one LIF step (accepted when idle)
//   busy                high from the accepting edge until clear_done / step_done
//   step_done           ONE-clock registered pulse; when seen, all 256 membranes are
//                       written and all 256 spike bits are captured
//   cur1_raddr/rdata    synchronous read of the cur1 cache (raw 35 bit, interpreted
//                       with $signed here)
//   hidden_spikes[255:0]  spikes of the most recently completed step (cleared by
//                       reset and by clear_state)
// Commands while busy are ignored. If clear_state and start_step arrive together
// while idle, clear_state wins.
//
// Reset resets the controller and hidden_spikes only; mem1 is a RAM and is NOT
// cleared by reset. The top level must issue clear_state before the first step of
// every new inference. Membranes persist between steps: clear only once per image.
//
// Schedule (2 clocks per neuron, E0 = edge that accepts the command)
//   step : E0  IDLE -> ISSUE, nidx = 0
//          E1  ISSUE: RAMs capture address j (cur1_raddr = mem1 raddr = nidx);  -> UPDATE
//          E2  UPDATE: cur1[j]/mem1[j] valid on the RAM outputs, lif_core is
//              combinational on them; mem1[j] <= membrane_new, hidden_spikes[j] <=
//              spike, nidx <= j+1 (the old nidx is used by this write)  -> ISSUE
//          neuron j is written at edge E(2j+2); neuron 255 at E512, where step_done
//          is registered together with the last write => latency 512 clocks.
//   clear: E0 IDLE -> CLEAR; edge E(j+1) writes mem1[j] = 0; E256 writes mem1[255]
//          and registers clear_done => latency 256 clocks.
// -----------------------------------------------------------------------------
module l1_lif_sweep
  import snn_params_pkg::*;
(
    input  logic                           clk,
    input  logic                           rst,

    input  logic                           clear_state,
    output logic                           clear_done,
    input  logic                           start_step,
    output logic                           busy,
    output logic                           step_done,

    output logic [7:0]                     cur1_raddr,
    input  logic [L1_ACC_WIDTH-1:0]        cur1_rdata,

    output logic [HIDDEN_SIZE-1:0]         hidden_spikes
);

  typedef enum logic [1:0] {S_IDLE, S_CLEAR, S_ISSUE, S_UPD} state_t;
  state_t state;

  logic [7:0] nidx;
  logic       last;
  assign last = (nidx == 8'(HIDDEN_SIZE - 1));
  assign busy = (state != S_IDLE);

  // ---- membrane RAM -----------------------------------------------------------
  logic                    mem_we;
  logic [7:0]              mem_waddr;
  logic [L1_MEM_WIDTH-1:0] mem_wdata, mem_rdata;

  mem1_ram #(.WIDTH(L1_MEM_WIDTH), .DEPTH(HIDDEN_SIZE), .ADDR_WIDTH(8)) u_mem1 (
      .clk(clk), .we(mem_we), .waddr(mem_waddr), .wdata(mem_wdata),
      .raddr(nidx), .rdata(mem_rdata));

  // ---- LIF update (verified block, combinational on the registered RAM outputs) ---
  logic signed [L1_MEM_WIDTH-1:0] lif_mem_new;
  logic                           lif_spike;

  lif_core #(.MEM_WIDTH(L1_MEM_WIDTH), .CUR_WIDTH(L1_ACC_WIDTH), .THRESHOLD(L1_THRESHOLD)) u_lif (
      .membrane_in($signed(mem_rdata)),
      .current_in($signed(cur1_rdata)),
      .membrane_out(lif_mem_new),
      .spike_out(lif_spike));

  assign cur1_raddr = nidx;
  assign mem_we     = (state == S_CLEAR) || (state == S_UPD);
  assign mem_waddr  = nidx;
  assign mem_wdata  = (state == S_CLEAR) ? '0 : lif_mem_new;

  // ---- controller ---------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      state         <= S_IDLE;
      nidx          <= '0;
      clear_done    <= 1'b0;
      step_done     <= 1'b0;
      hidden_spikes <= '0;
    end else begin
      clear_done <= 1'b0;
      step_done  <= 1'b0;
      case (state)
        S_IDLE: begin
          if (clear_state) begin
            nidx          <= '0;
            hidden_spikes <= '0;
            state         <= S_CLEAR;
          end else if (start_step) begin
            nidx  <= '0;
            state <= S_ISSUE;
          end
        end
        S_CLEAR: begin
          if (last) begin
            state      <= S_IDLE;
            clear_done <= 1'b1;
          end else begin
            nidx <= nidx + 8'd1;
          end
        end
        S_ISSUE: state <= S_UPD;
        S_UPD: begin
          hidden_spikes[nidx] <= lif_spike;
          if (last) begin
            state     <= S_IDLE;
            step_done <= 1'b1;
          end else begin
            nidx  <= nidx + 8'd1;
            state <= S_ISSUE;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
