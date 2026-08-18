`timescale 1ns / 1ps

module bdh_zero_skipper (
    input  wire signed [7:0] act,
    input  wire              sparse_mode,
    output wire              skip_cycle
);
    assign skip_cycle = sparse_mode && (act == 8'sd0);
endmodule
