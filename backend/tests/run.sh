export PYTHONPATH='.'
# 
file=tests/test_agent2.py
python $file \
    2>&1 | tee $file.log