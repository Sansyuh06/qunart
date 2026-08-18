`timescale 1ns / 1ps

module top #(
    parameter integer CLK_FREQ = 27000000,
    parameter integer BAUD     = 115200
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led
);
    wire [7:0] rx_data;
    wire       rx_valid;
    wire [7:0] tx_data;
    wire       tx_start;
    wire       tx_ready;
    wire       tx_busy;

    wire               mac_clear;
    wire               mac_enable;
    wire signed [7:0]  mac_weight;
    wire signed [7:0]  mac_act;
    wire signed [31:0] acc;
    wire               overflow;
    wire               skip_cycle;

    wire command_pulse;
    wire dense_stream_active;
    wire sparse_stream_active;
    wire tx_active;
    wire sparse_mode;


    uart_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD(BAUD)
    ) uart_rx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .rx(uart_rx),
        .data(rx_data),
        .valid(rx_valid)
    );

    uart_tx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD(BAUD)
    ) uart_tx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(tx_start),
        .data(tx_data),
        .tx(uart_tx),
        .ready(tx_ready),
        .busy(tx_busy)
    );

    bdh_mac_dsp mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .clear(mac_clear),
        .enable(mac_enable),
        .weight(mac_weight),
        .act(mac_act),
        .acc(acc),
        .overflow(overflow)
    );

    bdh_zero_skipper zero_skipper_inst (
        .act(mac_act),
        .sparse_mode(sparse_mode),
        .skip_cycle(skip_cycle)
    );

    bdh_fsm fsm_inst (
        .clk(clk),
        .rst_n(rst_n),
        .rx_data(rx_data),
        .rx_valid(rx_valid),
        .tx_data(tx_data),
        .tx_start(tx_start),
        .tx_ready(tx_ready),
        .acc(acc),
        .mac_clear(mac_clear),
        .mac_enable(mac_enable),
        .mac_weight(mac_weight),
        .mac_act(mac_act),
        .mac_overflow(overflow),
        .skip_cycle(skip_cycle),
        .command_pulse(command_pulse),
        .dense_stream_active(dense_stream_active),
        .sparse_stream_active(sparse_stream_active),
        .tx_active(tx_active),
        .sparse_mode(sparse_mode),
        .overflow_error()
    );

    bdh_led_chaser #(
        .CLK_FREQ(CLK_FREQ)
    ) led_chaser_inst (
        .sys_clk(clk),
        .sys_rst_n(rst_n),
        .processing(mac_enable | command_pulse | dense_stream_active | sparse_stream_active | tx_active),
        .led(led)
    );
endmodule
