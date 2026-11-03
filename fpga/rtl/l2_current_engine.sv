// -----------------------------------------------------------------------------
// l2_current_engine : all ten Layer-2 currents for one hidden-spike vector
// (RTL milestone 8):
//     cur2[k] = bias2[k] + sum_{j=0..255} hidden_spikes[j] * weight2[k][j]
// computed sequentially with ONE l2_acc (no multiplier, one accumulator).
//
// Interface
//   clk, rst           clock, synchronous active-high reset
//   start              accepted only when busy = 0 (ignored while busy)
//   hidden_spikes      256-bit vector, must be stable from start until done
//   busy               1 from the accepting edge until the done pulse
//   done               ONE-clock registered pulse; when seen, cur2[0..9] are final
//   cur2[0..9]         signed L2_ACC_WIDTH (33) registers; written once per run
//                      (not reset: a complete run overwrites all ten before done)
// Reset clears the controller, the index registers and the accumulator.
//
// Addressing (row-major [10][256], never transposed)
//   weight address = {k[3:0], j[7:0]} = k*256 + j   (concatenation, no multiplier)
//   bias address   = k
//
// Timeline for ONE output neuron k  (E0 = edge that accepts start; neuron k > 0
// repeats the same pattern shifted by 259 clocks per neuron)
//   E0   IDLE -> BIAS, k=0, j=0
//   E1   BIAS : bias ROM captures bias[k]; weight ROM captures {k,0} (unused). -> LOAD
//   E2   LOAD : l2_acc.load_bias (bias valid since E1). ROM captures address {k,0}
//               and spk_d <= hidden_spikes[0]; v1 <= 1; j <= 1. -> RUN
//   E3   RUN  : acc consumes index 0 (v1=1: weight[k][0] and spk_d on the outputs);
//               ROM captures address {k,1}, spk_d <= hidden_spikes[1].
//   E(2+j) RUN: ROM captures {k,j} and spk_d <= hidden_spikes[j]   (j = 1..255)
//   E(3+j) acc consumes index j (weight2[k][j] with the SAME j's spike)
//   E257 RUN  : captures {k,255}, spk_d <= hidden_spikes[255], j == 255 -> DRAIN
//   E258 DRAIN: acc consumes index 255 -> acc_out final after E258 (v1 <= 0)
//   E259 STORE: cur2[k] <= acc_out (final). k < 9: k <= k+1, j <= 0, -> BIAS
//               (next neuron's E1 is E260); k = 9: done <= 1, -> IDLE
//   => 259 clocks per output neuron = 1 bias ROM + 1 bias load + 255 RUN
//      + 1 drain + 1 store, plus the 256th index overlapped; total latency from
//      the accepting edge to done visible = 10*259 = 2590 clocks.
// The ROM output is registered, so each spike is delayed with the same registers
// (spk_d, v1) as the ROM data: the spike consumed with weight2[k][j] is always
// hidden_spikes[j], never hidden_spikes of the already-advanced index.
// -----------------------------------------------------------------------------
module l2_current_engine
  import snn_params_pkg::*;
(
    input  logic                           clk,
    input  logic                           rst,

    input  logic                           start,
    input  logic [HIDDEN_SIZE-1:0]         hidden_spikes,

    output logic                           busy,
    output logic                           done,
    output logic signed [L2_ACC_WIDTH-1:0] cur2 [0:OUTPUT_SIZE-1]
);

  initial begin
    if (HIDDEN_SIZE != 256) $fatal(1, "l2_current_engine: address concatenation assumes HIDDEN_SIZE = 256");
    if (OUTPUT_SIZE > 10)   $fatal(1, "l2_current_engine: k is 4 bits, max 16 outputs");
  end

  typedef enum logic [2:0] {S_IDLE, S_BIAS, S_LOAD, S_RUN, S_DRAIN, S_STORE} state_t;
  state_t state;

  logic [3:0] kidx;
  logic [7:0] jidx;
  logic       v1;       // ROM outputs hold weight[k][j] this cycle
  logic       spk_d;    // hidden_spikes[j] delayed with the ROM data

  assign busy = (state != S_IDLE);

  logic last_issue, last_neuron;
  assign last_issue  = (jidx == 8'(HIDDEN_SIZE - 1));
  assign last_neuron = (kidx == 4'(OUTPUT_SIZE - 1));

  logic signed [L2_ACC_WIDTH-1:0] acc_out;

  always_ff @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE;
      kidx  <= '0;
      jidx  <= '0;
      v1    <= 1'b0;
      spk_d <= 1'b0;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          v1 <= 1'b0;
          if (start) begin
            kidx  <= '0;
            jidx  <= '0;
            state <= S_BIAS;
          end
        end
        S_BIAS: state <= S_LOAD;
        S_LOAD: begin
          v1    <= 1'b1;
          spk_d <= hidden_spikes[jidx];
          jidx  <= jidx + 8'd1;
          state <= S_RUN;
        end
        S_RUN: begin
          v1    <= 1'b1;
          spk_d <= hidden_spikes[jidx];
          if (last_issue) state <= S_DRAIN;
          else            jidx  <= jidx + 8'd1;
        end
        S_DRAIN: begin
          v1    <= 1'b0;
          state <= S_STORE;
        end
        S_STORE: begin
          if (last_neuron) begin
            done  <= 1'b1;
            state <= S_IDLE;
          end else begin
            kidx  <= kidx + 4'd1;
            jidx  <= '0;
            state <= S_BIAS;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- result registers (written at STORE only) -------------------------------------
  always_ff @(posedge clk) begin
    if (state == S_STORE) cur2[kidx] <= acc_out;
  end

  // ---- ROMs (synchronous reads) -------------------------------------------------------
  logic [L2_WEIGHT_ADDR_WIDTH-1:0] w_raddr;
  logic [15:0] w_data;
  logic [31:0] bias_data;
  assign w_raddr = {kidx, jidx};

  sync_rom #(.WIDTH(WEIGHT_WIDTH), .DEPTH(L2_WEIGHT_DEPTH), .ADDR_WIDTH(L2_WEIGHT_ADDR_WIDTH),
             .INIT_FILE(L2_WEIGHT_FILE)) u_w2 (.clk(clk), .addr(w_raddr), .data(w_data));

  sync_rom #(.WIDTH(BIAS_WIDTH), .DEPTH(L2_BIAS_DEPTH), .ADDR_WIDTH(L2_BIAS_ADDR_WIDTH),
             .INIT_FILE(L2_BIAS_FILE)) u_b2 (.clk(clk), .addr(kidx), .data(bias_data));

  // ---- accumulator ------------------------------------------------------------------------
  logic acc_result_valid;
  l2_acc #(.ACC_WIDTH(L2_ACC_WIDTH), .N_INPUTS(HIDDEN_SIZE)) u_acc (
      .clk(clk), .rst(rst),
      .load_bias(state == S_LOAD), .bias($signed(bias_data)),
      .valid(v1), .spike(spk_d), .weight($signed(w_data)),
      .acc_out(acc_out), .result_valid(acc_result_valid));

endmodule
