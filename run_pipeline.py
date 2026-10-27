import argparse
from pathlib import Path
from src.pipeline import run_all

def main():
    parser = argparse.ArgumentParser(description="Pipeline de churn y segmentación.")
    parser.add_argument("--data", default="data/clientes.csv", help="Ruta del CSV de clientes.")
    parser.add_argument("--out", default="outputs", help="Carpeta de salida.")
    args = parser.parse_args()
    run_all(Path(args.data), Path(args.out))

if __name__ == "__main__":
    main()
