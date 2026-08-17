`timescale 1ns / 1ps

module bdh_fsm (
    input  wire               clk,
    input  wire               rst_n,
    input  wire [7:0]         rx_data,
    input  wire               rx_valid,
    output reg  [7:0]         tx_data,
    output reg                tx_start,
    input  wire               tx_ready,
    input  wire signed [31:0] acc,
    output reg                mac_clear,
    output reg                mac_enable,
    output reg signed [7:0]   mac_weight,
    output reg signed [7:0]   mac_act,
    input  wire               mac_overflow,
    input  wire               skip_cycle,
    output reg                command_pulse,
    output reg                dense_stream_active,
    output reg                sparse_stream_active,
    output reg                tx_active,
    output reg                sparse_mode,
    output wire               overflow_error
);
    localparam [7:0] CMD_RESET_ACC  = 8'h00;
    localparam [7:0] CMD_MAC_PAIR   = 8'h01;
    localparam [7:0] CMD_MAC_STREAM = 8'h02;
    localparam [7:0] CMD_READ_ACC   = 8'h03;
    localparam [7:0] CMD_MAC_SPARSE = 8'h04;

    localparam [3:0] S_IDLE              = 4'd0;
    localparam [3:0] S_DECODE_CMD        = 4'd1;
    localparam [3:0] S_RX_PAIR_WEIGHT    = 4'd2;
    localparam [3:0] S_RX_PAIR_ACT       = 4'd3;
    localparam [3:0] S_RX_STREAM_N0      = 4'd4;
    localparam [3:0] S_RX_STREAM_N1      = 4'd5;
    localparam [3:0] S_RX_STREAM_WEIGHT  = 4'd6;
    localparam [3:0] S_RX_STREAM_ACT     = 4'd7;
    localparam [3:0] S_TX_LOAD           = 4'd8;
    localparam [3:0] S_TX_WAIT_BUSY      = 4'd9;
    localparam [3:0] S_TX_WAIT_READY     = 4'd10;

    reg [3:0]  state;
    reg [15:0] stream_total;
    reg [15:0] stream_count;
    reg [31:0] tx_latched_acc;
    reg [1:0]  tx_index;
    reg [7:0]  command;
    reg        stream_is_sparse;

    assign overflow_error = mac_overflow;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                <= S_IDLE;
            stream_total         <= 16'd0;
            stream_count         <= 16'd0;
            tx_latched_acc       <= 32'd0;
            tx_index             <= 2'd0;
            tx_data              <= 8'd0;
            tx_start             <= 1'b0;
            command              <= 8'd0;
            stream_is_sparse     <= 1'b0;
            mac_clear            <= 1'b0;
            mac_enable           <= 1'b0;
            mac_weight           <= 8'sd0;
            mac_act              <= 8'sd0;
            command_pulse        <= 1'b0;
            dense_stream_active  <= 1'b0;
            sparse_stream_active <= 1'b0;
            tx_active            <= 1'b0;
            sparse_mode          <= 1'b0;
        end else begin
            tx_start      <= 1'b0;
            mac_clear     <= 1'b0;
            mac_enable    <= 1'b0;
            command_pulse <= 1'b0;

            case (state)
                S_IDLE: begin
                    dense_stream_active  <= 1'b0;
                    sparse_stream_active <= 1'b0;
                    tx_active            <= 1'b0;
                    sparse_mode          <= 1'b0;
                    if (rx_valid) begin
                        command       <= rx_data;
                        command_pulse <= 1'b1;
                        state         <= S_DECODE_CMD;
                    end
                end

                S_DECODE_CMD: begin
                    case (command)
                        CMD_RESET_ACC: begin
                            mac_clear <= 1'b1;
                            state     <= S_IDLE;
                        end
                        CMD_MAC_PAIR: begin
                            stream_is_sparse <= 1'b0;
                            sparse_mode      <= 1'b0;
                            state            <= S_RX_PAIR_WEIGHT;
                        end
                        CMD_MAC_STREAM: begin
                            stream_is_sparse <= 1'b0;
                            sparse_mode      <= 1'b0;
                            state            <= S_RX_STREAM_N0;
                        end
                        CMD_MAC_SPARSE: begin
                            stream_is_sparse <= 1'b1;
                            sparse_mode      <= 1'b1;
                            state            <= S_RX_STREAM_N0;
                        end
                        CMD_READ_ACC: begin
                            tx_latched_acc <= acc;
                            tx_index       <= 2'd0;
                            tx_active      <= 1'b1;
                            state          <= S_TX_LOAD;
                        end
                        default: begin
                            state <= S_IDLE;
                        end
                    endcase
                end

                S_RX_PAIR_WEIGHT: begin
                    if (rx_valid) begin
                        mac_weight <= rx_data;
                        state      <= S_RX_PAIR_ACT;
                    end
                end

                S_RX_PAIR_ACT: begin
                    if (rx_valid) begin
                        mac_act    <= rx_data;
                        mac_enable <= 1'b1;
                        state      <= S_IDLE;
                    end
                end

                S_RX_STREAM_N0: begin
                    if (rx_valid) begin
                        stream_total[7:0] <= rx_data;
                        state             <= S_RX_STREAM_N1;
                    end
                end

                S_RX_STREAM_N1: begin
                    if (rx_valid) begin
                        stream_total[15:8] <= rx_data;
                        stream_count       <= 16'd0;
                        if ((rx_data == 8'd0)
                                && (stream_total[7:0] == 8'd0)) begin
                            state <= S_IDLE;
                        end else begin
                            dense_stream_active  <= !stream_is_sparse;
                            sparse_stream_active <= stream_is_sparse;
                            sparse_mode          <= stream_is_sparse;
                            state                <= S_RX_STREAM_WEIGHT;
                        end
                    end
                end

                S_RX_STREAM_WEIGHT: begin
                    dense_stream_active  <= !stream_is_sparse;
                    sparse_stream_active <= stream_is_sparse;
                    sparse_mode          <= stream_is_sparse;
                    if (rx_valid) begin
                        mac_weight <= rx_data;
                        state      <= S_RX_STREAM_ACT;
                    end
                end

                S_RX_STREAM_ACT: begin
                    dense_stream_active  <= !stream_is_sparse;
                    sparse_stream_active <= stream_is_sparse;
                    sparse_mode          <= stream_is_sparse;
                    if (rx_valid) begin
                        mac_act      <= rx_data;
                        mac_enable   <= !skip_cycle;
                        stream_count <= stream_count + 16'd1;
                        if (stream_count + 16'd1 >= stream_total) begin
                            dense_stream_active  <= 1'b0;
                            sparse_stream_active <= 1'b0;
                            sparse_mode          <= 1'b0;
                            state                <= S_IDLE;
                        end else begin
                            state <= S_RX_STREAM_WEIGHT;
                        end
                    end
                end

                S_TX_LOAD: begin
                    tx_active <= 1'b1;
                    if (tx_ready) begin
                        case (tx_index)
                            2'd0: tx_data <= tx_latched_acc[7:0];
                            2'd1: tx_data <= tx_latched_acc[15:8];
                            2'd2: tx_data <= tx_latched_acc[23:16];
                            2'd3: tx_data <= tx_latched_acc[31:24];
                        endcase
                        tx_start <= 1'b1;
                        state    <= S_TX_WAIT_BUSY;
                    end
                end

                S_TX_WAIT_BUSY: begin
                    tx_active <= 1'b1;
                    if (!tx_ready) begin
                        state <= S_TX_WAIT_READY;
                    end
                end

                S_TX_WAIT_READY: begin
                    tx_active <= 1'b1;
                    if (tx_ready) begin
                        if (tx_index == 2'd3) begin
                            tx_active <= 1'b0;
                            state     <= S_IDLE;
                        end else begin
                            tx_index <= tx_index + 2'd1;
                            state    <= S_TX_LOAD;
                        end
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
