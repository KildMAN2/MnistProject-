// -----------------------------------------------------------------------------
// l1_neuron_engine : computes the layer-1 current of ONE selected hidden neuron
// (RTL milestone 5):
//     current_out = bias1[j] + sum_{i=0..783} pixel[i] * weight1[j][i]
//
// Interface
//   clk, rst           clock, synchronous active-high reset
//   pixel_we/waddr/wdata   write port of the pixel buffer (load image before start)
//   start, neuron_index    start is accepted only when busy = 0; neuron_index 0..255
//                          is captured at the accepting edge
//   busy               1 from the accepting edge until the done pulse
//   done               ONE-clock pulse; current_out is valid while done = 1
//   current_out        signed L1_ACC_WIDTH (35) bits, = l1_mac accumulator
//
// Addressing (row-major [256][784], never transposed)
//   weight address = neuron*784 + i.   The base is computed once at the accepting
//   edge as (j<<9)+(j<<8)+(j<<4) after ZERO-EXTENDING j to the 18-bit address
//   width (784 = 512+256+16), then a running address increments by 1 per pixel.
//   No multiplier. Bias address = neuron_index (captured register).
//   Pixel address = i.
//
// Timeline (E0 = clock edge that accepts start; E<k> = k-th edge after it)
//   E0   state IDLE->BIAS : nidx, w_raddr=base, pix_raddr=0 registered. busy=1
//   E1   bias ROM captures bias[nidx]; ROMs also capture addr 0 (not used yet)
//        state BIAS->LOAD
//   E2   state LOAD: l1_mac.load_bias=1 (bias data valid since E1, stable);
//        ROMs capture address 0 again -> pixel[0], weight[j][0] valid after E2;
//        v1<=1 ; addresses advance to 1
//   E(2+i) RUN: ROMs capture address i, v1 <= 1 (data i valid after the edge)
//   E(3+i) l1_mac.valid = v1 consumes pixel[i], weight[j][i]  (acc += product i)
//   E785 capture address 783, v1_last<=1, state RUN->DRAIN
//   E786 mac consumes product 783 -> acc_out complete after E786;
//        done<=1, busy->0 (DRAIN->IDLE).  done and the final current_out are
//        both visible right after E786.
//   => accepted-start edge to done = 786 clocks = 1 (bias ROM) + 1 (bias load)
//      + 784 (MAC products, one per clock; address issue is overlapped)
// valid is the address-capture event delayed by registers: v1 is set at the edge
// where the ROMs capture an address, so it is high exactly when that address's
// pixel/weight are on the ROM outputs.
// -----------------------------------------------------------------------------
module l1_neuron_engine
  import snn_params_pkg::*;
(
    input  logic                           clk,
    input  logic                           rst,

    input  logic                           pixel_we,
    input  logic [9:0]                     pixel_waddr,
    input  logic [7:0]                     pixel_wdata,

    input  logic                           start,
    input  logic [7:0]                     neuron_index,

    output logic                           busy,
    output logic                           done,
    output logic signed [L1_ACC_WIDTH-1:0] current_out
);

  localparam int AW = L1_WEIGHT_ADDR_WIDTH;
  localparam logic [9:0] LAST_PIXEL = 10'(INPUT_SIZE - 1);

  initial begin
    if (INPUT_SIZE != 784) $fatal(1, "l1_neuron_engine: shift/add base assumes INPUT_SIZE = 784");
    if (HIDDEN_SIZE > 256) $fatal(1, "l1_neuron_engine: neuron_index is 8 bits");
  end

  typedef enum logic [2:0] {S_IDLE, S_BIAS, S_LOAD, S_RUN, S_DRAIN} state_t;
  state_t state;

  logic [7:0]  nidx;
  logic [9:0]  pix_raddr;
  logic [AW-1:0] w_raddr;
  logic        v1;       // ROM outputs hold a valid pixel/weight pair this cycle
  logic        v1_last;  // ... and it is pixel 783

  // j*784 = (j<<9)+(j<<8)+(j<<4), zero-extended BEFORE shifting
  logic [AW-1:0] j_ext, base_addr;
  assign j_ext     = AW'(neuron_index);
  assign base_addr = (j_ext << 9) + (j_ext << 8) + (j_ext << 4);

  logic accept;
  assign accept = start && (state == S_IDLE);
  assign busy   = (state != S_IDLE);

  logic last_issue;
  assign last_issue = (pix_raddr == LAST_PIXEL);

  always_ff @(posedge clk) begin
    if (rst) begin
      state     <= S_IDLE;
      nidx      <= '0;
      pix_raddr <= '0;
      w_raddr   <= '0;
      v1        <= 1'b0;
      v1_last   <= 1'b0;
      done      <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          v1      <= 1'b0;
          v1_last <= 1'b0;
          if (accept) begin
            nidx      <= neuron_index;
            pix_raddr <= '0;
            w_raddr   <= base_addr;
            state     <= S_BIAS;
          end
        end
        S_BIAS: begin
          state <= S_LOAD;
        end
        S_LOAD: begin
          v1        <= 1'b1;
          v1_last   <= 1'b0;
          pix_raddr <= pix_raddr + 10'd1;
          w_raddr   <= w_raddr + AW'(1);
          state     <= S_RUN;
        end
        S_RUN: begin
          v1      <= 1'b1;
          v1_last <= last_issue;
          if (last_issue) begin
            state <= S_DRAIN;
          end else begin
            pix_raddr <= pix_raddr + 10'd1;
            w_raddr   <= w_raddr + AW'(1);
          end
        end
        S_DRAIN: begin
          v1      <= 1'b0;
          v1_last <= 1'b0;
          done    <= 1'b1;
          state   <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- memories (synchronous reads) -------------------------------------------
  logic [7:0]  pix_data;
  logic [15:0] w_data;
  logic [31:0] bias_data;

  pixel_buffer #(.WIDTH(8), .DEPTH(INPUT_SIZE), .ADDR_WIDTH(10)) u_pixels (
      .clk(clk), .write_enable(pixel_we), .write_addr(pixel_waddr), .write_data(pixel_wdata),
      .read_addr(pix_raddr), .read_data(pix_data));

  sync_rom #(.WIDTH(WEIGHT_WIDTH), .DEPTH(L1_WEIGHT_DEPTH), .ADDR_WIDTH(AW),
             .INIT_FILE(L1_WEIGHT_FILE)) u_w1 (.clk(clk), .addr(w_raddr), .data(w_data));

  sync_rom #(.WIDTH(BIAS_WIDTH), .DEPTH(L1_BIAS_DEPTH), .ADDR_WIDTH(L1_BIAS_ADDR_WIDTH),
             .INIT_FILE(L1_BIAS_FILE)) u_b1 (.clk(clk), .addr(nidx), .data(bias_data));

  // ---- MAC ------------------------------------------------------------------------
  logic mac_load_bias, mac_valid;
  assign mac_load_bias = (state == S_LOAD);
  assign mac_valid     = v1;

  logic mac_result_valid;
  l1_mac #(.ACC_WIDTH(L1_ACC_WIDTH)) u_mac (
      .clk(clk), .rst(rst),
      .load_bias(mac_load_bias), .bias($signed(bias_data)),
      .valid(mac_valid), .pixel(pix_data), .weight($signed(w_data)),
      .acc_out(current_out), .result_valid(mac_result_valid));

endmodule
