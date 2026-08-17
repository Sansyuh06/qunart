"""Wrapper to run pytest with increased stack size on Windows (Python 3.12 stack overflow workaround)."""
import sys
import threading

def run_tests():
    import pytest
    sys.exit(pytest.main(['-v', '--tb=short'] + sys.argv[1:]))

# 8MB stack to handle the deep import chains (transformers -> sklearn -> pandas)
thread = threading.Thread(target=run_tests)
thread.daemon = True
threading.stack_size(8 * 1024 * 1024)
thread = threading.Thread(target=run_tests)
thread.start()
thread.join()
