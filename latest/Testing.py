from latest import *
from latest.Helper import compare_to_exact
from latest.TestCases import *
import csv
import os

def test_from_str(char: dict, name: str, enewick_str: str = None, network: PhylogeneticNetwork = None):
    if network is None:
        network = enewick_to_network(enewick_str, False)
    return compare_to_exact(network, char, f"{name}_network.txt", f"{name}_character.txt")

csv_path     = "results.csv"
field_order  = [
    "vertices", "id", "retic_ratio",
    "pre_result", "post_result", "exact_result",
    "pre_ratio", "post_ratio",
    "algo_time", "exact_time",
]

header_needed = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
csv_file      = open(csv_path, "a", newline="")
writer        = csv.DictWriter(csv_file, fieldnames=field_order)
if header_needed:
    writer.writeheader()

def call_all():
    test = TestCases()
    for id in range(1, 1000000):
        vertices = random.randint(5, 5000)
        rate1 = rate2 = rate3 = 0.5

        network, char, retic_ratio = test.random_network(
            vertices, rate1, rate2, rate3
        )
        pre_res, post_res, exact_res, algo_time, exact_time = test_from_str(
            char,
            f"random_network_{vertices}_{id}_",
            network=network,
        )
        # compute derived ratios
        pre_ratio  = pre_res  / exact_res if exact_res else None
        post_ratio = post_res / exact_res if exact_res else None

        if pre_ratio and pre_ratio > 2:
            print(f"random_network_{vertices}_{id}")

        # stream one row straight to CSV
        writer.writerow({
            "vertices":     vertices,
            "id": id,
            "retic_ratio":  retic_ratio,
            "pre_result":   pre_res,
            "post_result":  post_res,
            "exact_result": exact_res,
            "pre_ratio":    pre_ratio,
            "post_ratio":   post_ratio,
            "algo_time":    algo_time,
            "exact_time":   exact_time,
        })
        csv_file.flush()            # make sure it’s on disk right away

call_all()
# csv_file.close()
