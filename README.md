The script in this repository helps to measure mlp/mha cases from [this paper](https://arxiv.org/pdf/2301.01333) on [oneDNN graph compiler](https://github.com/oneapi-src/oneDNN/tree/main/src/graph/backend/graph_compiler) with/without coarse-grained fusion using [benchdnn](https://github.com/oneapi-src/oneDNN/tree/main/tests/benchdnn#benchdnn).

The `cases/` folder contains deserialized graphs describing mlp/mha patterns from the paper.

In order to run the script:
1. Apply `enable_fusion_control.patch` to oneDNN in order to being able enable/disable coarse grained fusion using env variable.
2. Install requirements for the script `pip install -r requirements.txt` (`pandas` and `openpyxl` to write the result to excel)
3. Build oneDNN with benchdnn and graph compiler enabled
4. Set `BENCHDNN_PATH` env variable to the `benchdnn` executable
5. (optional) Adjust `test_files` variable in the script to select desired subset of tests
6. (optional) Adjust `OMP_NUM_THREADS` variable in the script
7. Run the script, it will generate an excel file with the measurements
