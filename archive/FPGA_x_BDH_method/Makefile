# ==============================================================================
# Makefile for Tang Nano 9K BDH Hardware Accelerator
# ==============================================================================
DEVICE = GW1NR-LV9QN88PC6/I5
FAMILY = GW1N-9C
BOARD  = tangnano9k

VERILOG_FILES = rtl/top.v rtl/uart_rx.v rtl/uart_tx.v rtl/bdh_fsm.v rtl/bdh_mac_dsp.v rtl/bdh_zero_skipper.v rtl/bdh_led_chaser.v
CST_FILE      = constraints/tang_nano_9k.cst

.PHONY: all synth pnr bitstream flash test-bdh lint-python clean

all: bitstream

build:
	python -c "from pathlib import Path; Path('build').mkdir(exist_ok=True)"

synth: build
	yosys -p "read_verilog $(VERILOG_FILES); synth_gowin -json build/bdh_accelerator.json"

pnr: synth
	nextpnr-gowin --json build/bdh_accelerator.json --write build/bdh_accelerator_pnr.json --device $(DEVICE) --family $(FAMILY) --cst $(CST_FILE)

bitstream: pnr
	gowin_pack -d $(FAMILY) -o build/bdh_accelerator.fs build/bdh_accelerator_pnr.json

flash:
	openFPGALoader -b $(BOARD) -f build/bdh_accelerator.fs

test-bdh:
	python python/test_bdh_hardware.py --port COM3 --baudrate 115200

lint-python:
	python -m flake8 python
	python -m mypy python
	python -m py_compile python/bdh_fpga_driver.py python/bdh_fpga_layer.py python/bdh_model.py python/test_bdh_hardware.py

clean:
	python -c "import shutil; from pathlib import Path; shutil.rmtree(Path('build'), ignore_errors=True)"
