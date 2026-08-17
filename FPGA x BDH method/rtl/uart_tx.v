`timescale 1ns / 1ps

module uart_tx #(
    parameter integer CLK_FREQ = 27000000,
    parameter integer BAUD     = 115200
) (
    input  wire      clk,
    input  wire      rst_n,
    input  wire      start,
    input  wire [7:0] data,
    output reg       tx,
    output wire      ready,
    output reg       busy
);
    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;

    localparam [1:0] S_IDLE  = 2'd0;
    localparam [1:0] S_START = 2'd1;
    localparam [1:0] S_DATA  = 2'd2;
    localparam [1:0] S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_count;
    reg [2:0]  bit_index;
    reg [7:0]  tx_shift;

    assign ready = (state == S_IDLE) && !busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            clk_count <= 16'd0;
            bit_index <= 3'd0;
            tx_shift  <= 8'd0;
            tx        <= 1'b1;
            busy      <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx        <= 1'b1;
                    busy      <= 1'b0;
                    clk_count <= 16'd0;
                    bit_index <= 3'd0;
                    if (start) begin
                        tx_shift <= data;
                        busy     <= 1'b1;
                        state    <= S_START;
                    end
                end

                S_START: begin
                    tx   <= 1'b0;
                    busy <= 1'b1;
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 16'd0;
                        state     <= S_DATA;
                    end else begin
                        clk_count <= clk_count + 16'd1;
                    end
                end

                S_DATA: begin
                    tx   <= tx_shift[bit_index];
                    busy <= 1'b1;
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 16'd0;
                        if (bit_index == 3'd7) begin
                            bit_index <= 3'd0;
                            state     <= S_STOP;
                        end else begin
                            bit_index <= bit_index + 3'd1;
                        end
                    end else begin
                        clk_count <= clk_count + 16'd1;
                    end
                end

                S_STOP: begin
                    tx   <= 1'b1;
                    busy <= 1'b1;
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 16'd0;
                        state     <= S_IDLE;
                    end else begin
                        clk_count <= clk_count + 16'd1;
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
