`timescale 1ns / 1ps

module led (
    input  wire       sys_clk,
    input  wire       sys_rst_n,
    output reg  [5:0] led
);

    // 27 MHz clock, 10-bit PWM (1024 brightness levels)
    reg [9:0]  pwm_counter;

    // Controls step speed — tune this to set overall glide speed
    reg [12:0] move_counter;
    localparam [12:0] MOVE_DIV = 13'd3374;   // ~8000 steps/sec @ 27MHz -> smooth crawl

    reg [2:0] position;      // which LED pair is currently active
    reg [9:0] fade;          // 0 -> 1023, one step at a time

    // Gamma-corrected (squared) brightness levels, computed combinationally
    wire [19:0] fade_sq   = fade * fade;                 // up to 20 bits
    wire [19:0] fade_inv  = (10'd1023 - fade);
    wire [19:0] inv_sq    = fade_inv * fade_inv;

    wire [9:0] br_in  = fade_sq[19:10];   // incoming LED brightness (gamma corrected)
    wire [9:0] br_out = inv_sq[19:10];    // outgoing LED brightness (gamma corrected)

    always @(posedge sys_clk or negedge sys_rst_n) begin

        if (!sys_rst_n) begin
            pwm_counter  <= 10'd0;
            move_counter <= 13'd0;
            position     <= 3'd0;
            fade         <= 10'd0;
            led          <= 6'b111111;
        end

        else begin

            pwm_counter <= pwm_counter + 1'b1;

            // ------------------------------------------------
            // Smooth, fine-grained movement (1 step at a time)
            // ------------------------------------------------
            if (move_counter == MOVE_DIV) begin
                move_counter <= 13'd0;

                if (fade < 10'd1023) begin
                    fade <= fade + 10'd1;
                end
                else begin
                    fade <= 10'd0;
                    position <= (position == 3'd5) ? 3'd0 : position + 1'b1;
                end
            end
            else begin
                move_counter <= move_counter + 1'b1;
            end

            // ------------------------------------------------
            // ACTIVE LOW LED PWM, gamma-corrected crossfade
            // ------------------------------------------------
            led <= 6'b111111;

            case (position)
                3'd0: begin
                    if (pwm_counter < br_out) led[0] <= 1'b0;
                    if (pwm_counter < br_in)  led[1] <= 1'b0;
                end
                3'd1: begin
                    if (pwm_counter < br_out) led[1] <= 1'b0;
                    if (pwm_counter < br_in)  led[2] <= 1'b0;
                end
                3'd2: begin
                    if (pwm_counter < br_out) led[2] <= 1'b0;
                    if (pwm_counter < br_in)  led[3] <= 1'b0;
                end
                3'd3: begin
                    if (pwm_counter < br_out) led[3] <= 1'b0;
                    if (pwm_counter < br_in)  led[4] <= 1'b0;
                end
                3'd4: begin
                    if (pwm_counter < br_out) led[4] <= 1'b0;
                    if (pwm_counter < br_in)  led[5] <= 1'b0;
                end
                3'd5: begin
                    if (pwm_counter < br_out) led[5] <= 1'b0;
                    if (pwm_counter < br_in)  led[0] <= 1'b0;
                end
                default: led <= 6'b111111;
            endcase

        end
    end

endmodule
