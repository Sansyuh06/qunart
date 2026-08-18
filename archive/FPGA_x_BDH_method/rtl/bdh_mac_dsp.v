`timescale 1ns / 1ps

module bdh_mac_dsp (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear,
    input  wire               enable,
    input  wire signed [7:0]  weight,
    input  wire signed [7:0]  act,
    output reg  signed [31:0] acc,
    output reg                overflow
);
    wire signed [15:0] prod;
    wire signed [31:0] prod_ext;
    wire signed [31:0] next_acc;

    assign prod     = weight * act;
    assign prod_ext = {{16{prod[15]}}, prod};
    assign next_acc = acc + prod_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc      <= 32'sd0;
            overflow <= 1'b0;
        end else if (clear) begin
            acc      <= 32'sd0;
            overflow <= 1'b0;
        end else if (enable) begin
            acc <= next_acc;
            if ((acc[31] == prod_ext[31]) && (next_acc[31] != acc[31])) begin
                overflow <= 1'b1;
            end
        end
    end
endmodule
