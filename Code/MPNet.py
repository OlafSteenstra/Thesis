import subprocess
from pathlib import Path

CPLEX_HOME = Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212")
work_dir   = Path(r"C:\Users\jantj\Documents\mpnet")

def MPNet(network, character):
    java_cmd = [
        "java",
        "-Xmx8g",
        f"-Djava.library.path={CPLEX_HOME/'cplex/bin/x64_win64'}",
        "-cp", f".;{CPLEX_HOME/'cplex/lib/cplex.jar'}",
        "MPNet",
        network,
        character,
        "--softwired",
        "--nodot"
    ]

    try:
        result = subprocess.run(
            java_cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
            check=True
        )
    except subprocess.TimeoutExpired:
        print(f"❌  MPNet did not finish in 1h.")
        return None
    except subprocess.CalledProcessError as e:
        print("❌  MPNet crashed or was killed:")
        print(e.stderr)
        return None

    for line in result.stdout.splitlines():
        if "***** Softwired Parsimony Score:" in line:
            return int(line.split(":")[-1].strip())

    print("⚠️  Softwired score not found:")
    print(result.stdout)
    return None
