// -----------------------------------------------------------------------------
// l2_lif_sweep : ONE Layer-2 LIF time step for the 10 output neurons, with
// persistent membrane state (RTL milestone 9).
//
// For k = 0..9 (strictly in this order, ONE shared lif_core):
//   {mem_new, spike} = lif_core(mem2[k], cur2[k])      (verified block, unchanged)
//   mem2[k] <= mem_new ; output_spikes[k] <= spike
//
// Interface
//   clk, rst          clock, synchronous active-high reset
//   clear_state       pulse: sequentially write mem2[0..9] = 0 and output_spikes = 0
//   clear_done        ONE-clock registered pulse after the last zero write
//   start_step        pulse: run one LIF step (accepted when idle)
//   busy              high from the accepting edge until clear_done / step_done
//   step_done         ONE-clock registered pulse; when seen, mem2[0..9] and
//                     output_spikes[9:0] are final for this step
//   cur2[0:9]         signed L2_ACC_WIDTH (33) layer-2 currents, must be stable
//                     from start_step until step_done
//   output_spikes     spikes of the most recently completed step
// Commands while busy are ignored. If clear_state and start_step arrive together
// while idle, clear_state wins.
//
// mem2 is a plain 10 x 36 register array (360 flip-flops): no RAM is needed.
// RESET clears the controller, output_spikes AND mem2 (cheap for 360 flops). The
// top level must nevertheless issue clear_state at the start of every inference;
// membranes persist between steps and are cleared once per image only.
// mem2 is readable hierarchically (u_mem2 is not needed: dut.mem2[k]).
//
// Schedule (E0 = edge that accepts the command); lif_core is combinational on the
// register array, so no memory latency exists:
//   step : E0 IDLE -> UPD, k = 0
//          E(k+1): UPD writes mem2[k], output_spikes[k], k <= k+1
//          E10   : writes neuron 9, step_done registered at the same edge
//          => 10 clocks from the accepting edge to step_done visible
//   clear: E0 IDLE -> CLEAR; E(k+1) writes mem2[k] = 0; E10 writes mem2[9] and
//          registers clear_done => 10 clocks
// (measured values are confirmed by the testbench, see docs)
// -----------------------------------------------------------------------------
module l2_lif_sweep
  import snn_params_pkg::*;
(
    input  logic                           clk,
    input  logic                           rst,

    input  logic                           clear_state,
    output logic                           clear_done,
    input  logic                           start_step,
    output logic                           busy,
    output logic                           step_done,

    input  logic signed [L2_ACC_WIDTH-1:0] cur2 [0:OUTPUT_SIZE-1],

    output logic [OUTPUT_SIZE-1:0]         output_spikes
);

  typedef enum logic [1:0] {S_IDLE, S_CLEAR, S_UPD} state_t;
  state_t state;

  logic [3:0] kidx;
  logic       last;
  assign last = (kidx == 4'(OUTPUT_SIZE - 1));
  assign busy = (state != S_IDLE);

  logic signed [L2_MEM_WIDTH-1:0] mem2 [0:OUTPUT_SIZE-1];

  // ---- one shared LIF core ---------------------------------------------------------
  logic signed [L2_MEM_WIDTH-1:0] lif_mem_new;
  logic                           lif_spike;

  lif_core #(.MEM_WIDTH(L2_MEM_WIDTH), .CUR_WIDTH(L2_ACC_WIDTH), .THRESHOLD(L2_THRESHOLD)) u_lif (
      .membrane_in(mem2[kidx]),
      .current_in(cur2[kidx]),
      .membrane_out(lif_mem_new),
      .spike_out(lif_spike));

  always_ff @(posedge clk) begin
    if (rst) begin
      state         <= S_IDLE;
      kidx          <= '0;
      clear_done    <= 1'b0;
      step_done     <= 1'b0;
      output_spikes <= '0;
      for (int k = 0; k < OUTPUT_SIZE; k++) mem2[k] <= '0;
    end else begin
      clear_done <= 1'b0;
      step_done  <= 1'b0;
      case (state)
        S_IDLE: begin
          if (clear_state) begin
            kidx          <= '0;
            output_spikes <= '0;
            state         <= S_CLEAR;
          end else if (start_step) begin
            kidx  <= '0;
            state <= S_UPD;
          end
        end
        S_CLEAR: begin
          mem2[kidx] <= '0;
          if (last) begin
            state      <= S_IDLE;
            clear_done <= 1'b1;
          end else begin
            kidx <= kidx + 4'd1;
          end
        end
        S_UPD: begin
          mem2[kidx]          <= lif_mem_new;
          output_spikes[kidx] <= lif_spike;
          if (last) begin
            state     <= S_IDLE;
            step_done <= 1'b1;
          end else begin
            kidx <= kidx + 4'd1;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
