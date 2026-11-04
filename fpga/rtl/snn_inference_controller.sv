// -----------------------------------------------------------------------------
// snn_inference_controller : global 25-time-step inference sequencer
// (RTL milestone 12). CONTROL ONLY: no arithmetic, no memory, no latency counters.
// Every child block is driven by a one-clock start/clear pulse and the controller
// waits for the child's real done pulse (handshake).
//
// One inference (pixels are already loaded in the Layer-1 pixel buffer):
//   IDLE --start--> L1C_KICK  : l1_current_start  (cur1 computed ONCE, cached)
//                   L1C_WAIT  : wait l1_current_done
//                   CLR_KICK  : l1_clear_state + l2_clear_state + clear_counts (1 clock)
//                   CLR_WAIT  : wait BOTH l1_clear_done and l2_clear_done
//   for timestep = 0..24
//                   L1S_KICK  : l1_start_step           L1S_WAIT : wait l1_step_done
//                   L2C_KICK  : l2_current_start        L2C_WAIT : wait l2_current_done
//                   L2L_KICK  : l2_start_step           L2L_WAIT : wait l2_step_done
//                   COUNT     : count_enable = 1 for exactly one clock
//                               timestep != 24 : timestep <= timestep + 1, -> L1S_KICK
//                               timestep == 24 : -> FINALIZE
//   FINALIZE : the 25th count edge has already updated the counters, so the
//              combinational argmax now sees the FINAL counts. predicted_digit is
//              registered here (one edge AFTER the last COUNT edge), done <= 1, -> IDLE.
//              Capturing on the COUNT edge itself would register the argmax of the
//              counters BEFORE their nonblocking update (stale by one step).
//
// Interface
//   clk, rst                 synchronous active-high reset (abort: returns to IDLE, no done)
//   start                    accepted only when busy = 0, ignored while busy
//   busy                     high from the accepting edge until the cycle of done; low
//                            when done is visible
//   done                     ONE-clock registered pulse; predicted_digit is valid when seen
//   predicted_digit[3:0]     registered; stable after done until the next inference finishes
//   timestep[4:0]            0..24, increments only after the step has been counted
//   argmax_prediction        combinational argmax10 output (only sampled in FINALIZE)
//   *_start / *_clear_state / count_enable / clear_counts : one-clock pulses (Moore outputs)
//   *_done                   child done pulses
// -----------------------------------------------------------------------------
module snn_inference_controller
  import snn_params_pkg::*;
(
    input  logic        clk,
    input  logic        rst,
    input  logic        start,

    input  logic        l1_current_done,
    input  logic        l1_clear_done,
    input  logic        l1_step_done,
    input  logic        l2_clear_done,
    input  logic        l2_current_done,
    input  logic        l2_step_done,
    input  logic [3:0]  argmax_prediction,

    output logic        busy,
    output logic        done,
    output logic [3:0]  predicted_digit,
    output logic [4:0]  timestep,

    output logic        l1_current_start,
    output logic        l1_clear_state,
    output logic        l1_start_step,
    output logic        l2_clear_state,
    output logic        l2_current_start,
    output logic        l2_start_step,
    output logic        clear_counts,
    output logic        count_enable
);

  typedef enum logic [3:0] {
    S_IDLE, S_L1C_KICK, S_L1C_WAIT, S_CLR_KICK, S_CLR_WAIT,
    S_L1S_KICK, S_L1S_WAIT, S_L2C_KICK, S_L2C_WAIT, S_L2L_KICK, S_L2L_WAIT,
    S_COUNT, S_FINALIZE
  } state_t;
  state_t state;

  logic l1_clr_seen, l2_clr_seen;

  assign busy = (state != S_IDLE);

  assign l1_current_start = (state == S_L1C_KICK);
  assign l1_clear_state   = (state == S_CLR_KICK);
  assign l2_clear_state   = (state == S_CLR_KICK);
  assign clear_counts     = (state == S_CLR_KICK);
  assign l1_start_step    = (state == S_L1S_KICK);
  assign l2_current_start = (state == S_L2C_KICK);
  assign l2_start_step    = (state == S_L2L_KICK);
  assign count_enable     = (state == S_COUNT);

  always_ff @(posedge clk) begin
    if (rst) begin
      state           <= S_IDLE;
      done            <= 1'b0;
      predicted_digit <= 4'd0;
      timestep        <= 5'd0;
      l1_clr_seen     <= 1'b0;
      l2_clr_seen     <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            timestep <= 5'd0;
            state    <= S_L1C_KICK;
          end
        end

        S_L1C_KICK: state <= S_L1C_WAIT;
        S_L1C_WAIT: if (l1_current_done) state <= S_CLR_KICK;

        S_CLR_KICK: begin
          l1_clr_seen <= 1'b0;
          l2_clr_seen <= 1'b0;
          state       <= S_CLR_WAIT;
        end
        S_CLR_WAIT: begin
          // the two clear-done pulses arrive at different times; remember the first one
          if (l1_clear_done) l1_clr_seen <= 1'b1;
          if (l2_clear_done) l2_clr_seen <= 1'b1;
          if ((l1_clr_seen || l1_clear_done) && (l2_clr_seen || l2_clear_done))
            state <= S_L1S_KICK;
        end

        S_L1S_KICK: state <= S_L1S_WAIT;
        S_L1S_WAIT: if (l1_step_done) state <= S_L2C_KICK;

        S_L2C_KICK: state <= S_L2C_WAIT;
        S_L2C_WAIT: if (l2_current_done) state <= S_L2L_KICK;

        S_L2L_KICK: state <= S_L2L_WAIT;
        S_L2L_WAIT: if (l2_step_done) state <= S_COUNT;

        S_COUNT: begin
          if (timestep == 5'(NUM_STEPS - 1)) begin
            state <= S_FINALIZE;
          end else begin
            timestep <= timestep + 5'd1;
            state    <= S_L1S_KICK;
          end
        end

        S_FINALIZE: begin
          predicted_digit <= argmax_prediction;
          done            <= 1'b1;
          state           <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
