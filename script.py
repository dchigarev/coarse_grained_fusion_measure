import subprocess
import os
import re
import pandas
from timeit import default_timer as timer

benchdnn_path = os.environ.get("BENCHDNN_PATH", "../build/tests/benchdnn/benchdnn")
OMP_NUM_THREADS = "96"

# Array of paths for JSON files
test_files = [
    ## MLP cases
    "cases/mlp_1_b128_int8.json",
    "cases/mlp_1_b256_int8.json",
    "cases/mlp_1_b512_int8.json",
    "cases/mlp_1_b128_f32.json",
    "cases/mlp_1_b256_f32.json",
    "cases/mlp_1_b512_f32.json",
    "cases/mlp_2_b128_int8.json",
    "cases/mlp_2_b256_int8.json",
    "cases/mlp_2_b512_int8.json",
    "cases/mlp_2_b128_f32.json",
    "cases/mlp_2_b256_f32.json",
    "cases/mlp_2_b512_f32.json",

    ## MHA cases
    # "cases/mha_1_b32_int8.json",
    # "cases/mha_1_b64_int8.json",
    # "cases/mha_1_b128_int8.json",
    # "cases/mha_2_b32_int8.json",
    # "cases/mha_2_b64_int8.json",
    # "cases/mha_2_b128_int8.json",
    # "cases/mha_3_b32_int8.json",
    # "cases/mha_3_b64_int8.json",
    # "cases/mha_3_b128_int8.json",
    # "cases/mha_4_b32_int8.json",
    # "cases/mha_4_b64_int8.json",
    # "cases/mha_4_b128_int8.json",

    ## AxBxC cases
    # "cases/AxBxC_int8_8192x8192.json",
    # "cases/AxBxC_f16_8192x8192.json",
    # "cases/AxBxC_f32_8192x8192.json",
]
test_names = [
    os.path.splitext(os.path.basename(file_path))[0] for file_path in test_files
]

t0 = timer()

ONEDNN_FUSION_LEVEL = ["2", "3"]
NRUNS = 3

# Array to store the command outputs
outputs = []
pattern = r"min\(ms\):(\d+\.\d+)"

df = pandas.DataFrame(
    index=test_names,
    columns=pandas.MultiIndex.from_tuples(
        [
            ("RELATIVE SPEEDUP", "FUSED"),
            ("RELATIVE SPEEDUP", "NON-FUSED"),
            ("ABSOLUTE TIME (ms)", "FUSED"),
            ("ABSOLUTE TIME (ms)", "NON-FUSED"),
            ("ABSOLUTE TIME (ms)", "ABSOLUTE TIME DIFF (MS)"),
        ]
    ),
)
df.columns.names = [f"OMP_NUM_THREADS={OMP_NUM_THREADS}", None]
num_iters = len(test_files) * len(ONEDNN_FUSION_LEVEL)
it_num = 1

# Search for the pattern in the text
# Iterate over the test_cases array
for i, json_file in enumerate(test_files):
    # Construct the subprocess command
    command = f"{benchdnn_path} --mode=P --graph --case={json_file}"
    for fusion_level in ONEDNN_FUSION_LEVEL:
        min_times = []
        for _ in range(NRUNS):
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env={
                    "ONEDNN_FUSION_LEVEL": fusion_level,
                    "OMP_NUM_THREADS": OMP_NUM_THREADS,
                    **dict(os.environ),
                },
            )
            match = re.search(pattern, result.stdout)
            # Extract and print the value if found
            if match:
                min_ms_value = float(match.group(1))
                min_times.append(min_ms_value)
                # print("found value:", min_ms_value)
            else:
                breakpoint()
                raise RuntimeError(f"Can't parse output: {result.stdout}")
            outputs.append(result.stdout)

        case_time = min(min_times)
        print(f"OMP_NUM_THREADS={OMP_NUM_THREADS} FUSION_LEVEL={fusion_level} {test_names[i]}: {case_time}ms | test case {it_num}/{num_iters} ({round((it_num / num_iters) * 100, 2)}%)")
        df.loc[test_names[i], ("ABSOLUTE TIME (ms)", "FUSED" if fusion_level == "3" else "NON-FUSED")] = (
            case_time
        )
        it_num += 1

df[("ABSOLUTE TIME (ms)", "ABSOLUTE TIME DIFF (MS)")] = df[("ABSOLUTE TIME (ms)", "NON-FUSED")] - df[("ABSOLUTE TIME (ms)", "FUSED")]
df[("RELATIVE SPEEDUP", "NON-FUSED")] = 1
df[("RELATIVE SPEEDUP", "FUSED")] = df[("ABSOLUTE TIME (ms)", "NON-FUSED")] / df[("ABSOLUTE TIME (ms)", "FUSED")]

print(df)

fname = f"coarse_grained_fusion_{OMP_NUM_THREADS}.xlsx"
try:
    df.to_excel(fname)
except Exception as e:
    print(f"Can't save the result to a file because of: {e}.\nTry to save the 'df' object manually (using csv for example)")
    breakpoint()
print(f"written result to: {fname}")
print(f"time took: {round(timer() - t0, 2)}s")
