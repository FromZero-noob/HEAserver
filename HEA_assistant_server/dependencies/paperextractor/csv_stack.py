import os

import pandas as pd
from tqdm import tqdm


def concat_all_csv_data(pts,out_dir="merged_all.csv",axis=0):
    # 合并所有的csv文件
    merge = []
    for i in tqdm(pts, desc="Merging CSV files"):
        if not os.path.isfile(i):
            print(f"File {i} does not exist, skipping.")
            continue
        try:
            di = pd.read_csv(i, index_col=0, encoding="utf-8")
        except Exception as e:

            print(f"Error reading {i}: {e}")
        merge.append(di)

    data  = pd.concat(merge, axis=axis)
    data.to_csv(out_dir, index=True, encoding="utf-8")
    data.to_excel(out_dir.replace(".csv", ".xlsx"), index=True)

# if __name__ == "__main__":
#     pt = pathlib.Path(f"/dp-library/solid_electrolyte/filter04/filter_true.csv")

#     data = pd.read_csv(pt, index_col=0, encoding="utf-8")
#     pts = data["name"].tolist()[:1000]

#     ot = pathlib.Path("/root/paperextractor/soild_data_root")

#     pts = [ ot / f"{i}/0_raw_csv/{i}_end.csv" for i in pts]
#     out_dir = ot / "merged_all.csv"
#     concat_all_csv_data(pts, out_dir=out_dir)




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concatenate all CSV files.")
    parser = add_args(parser)
    args = parser.parse_args()
    run(args)


def add_args(parser):
    parser.add_argument("input_files", nargs='*', help="List of CSV files to concatenate.")
    parser.add_argument("-pf", "--paths_file", default=None, help="File containing paths to CSV files, one per line.")
    parser.add_argument("-o", "--output_file", default="merged_all.csv", help="Output file name for the merged CSV.")
    parser.add_argument("-c","--config", default=None, help="Configuration json file for additional settings.")
    return parser

def run(args):

    if args.config:
        import json
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file {args.config} does not exist.")
        with open(args.config, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    if args.paths_file:
        if not os.path.isfile(args.paths_file):
            raise FileNotFoundError(f"Paths file {args.paths_file} does not exist.")
        with open(args.paths_file, 'r') as f:
            args.input_files = [line.strip() for line in f if line.strip()]

    for k,v in args.__dict__.items():
        print(f"Argument {k} = {v}")

    concat_all_csv_data(args.input_files, out_dir=args.output_file)
    print(f"All CSV files have been merged into {args.output_file}")

if __name__ == "__main__":
    main()