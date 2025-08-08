import itertools
import json
import os
import pathlib
import re

import numpy as np
import requests

from paperextractor.utils.doi_to_name import name_to_doi
from paperextractor.utils.extract_figure.get_figure import (
    extract_pdf_region_to_png,
)


def trigger_file_original(file_path, inner=False,out_dir="./manuscript_figures",out_name="sub_figures_original.json"):
    if inner:
        host = "http://192.168.192.225:4002"
    else:
        host = "http://101.126.82.63:4002"

    url = f"{host}/trigger-file-async"
    result_url = f"{host}/get-result"

    files = {"file": open(file_path, "rb")}
    data = {
        "token": os.path.basename(file_path)[:15],
        "lang": "en",
    }
    semantic_cfg = {
        "textual": True,
        "chart": False,
        "table": False,
        "molecule": False,
        "equation": False,
        "figure": False,
        "expression": False,
    }
    data.update(semantic_cfg)
    print(data)

    sync = True
    response = requests.post(url, files=files, data={"sync": sync, **data}).json()
    # print(json.dumps(response, indent=4))
    if response["status"] != "success":
        return None

    data["pages_dict"] = True
    result = requests.post(result_url, json=data).json()
    if result["status"] != "success":
        return None

    pages_dict = result["pages_dict"]

    out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_filepath = os.path.join(out_dir, out_name)



    with open(out_filepath, "w") as f:
        json.dump(pages_dict, f, indent=4)

    return out_filepath

def compute_iof(group_bboxes: np.ndarray, other_bboxes: np.ndarray):
    """
    group_bboxes: [N, 4]
    other_bboxes: [M, 4]
    return: [N, M] IOF matrix
    """
    # N = group_bboxes.shape[0]
    # M = other_bboxes.shape[0]

    group = group_bboxes[:, None, :]  # [N, 1, 4]
    other = other_bboxes[None, :, :]  # [1, M, 4]

    # Intersection
    inter_x1 = np.maximum(group[..., 0], other[..., 0])
    inter_y1 = np.maximum(group[..., 1], other[..., 1])
    inter_x2 = np.minimum(group[..., 2], other[..., 2])
    inter_y2 = np.minimum(group[..., 3], other[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    intersection = inter_w * inter_h  # [N, M]

    # Foreground (other bboxes) area
    other_area = (other[..., 2] - other[..., 0]) * (other[..., 3] - other[..., 1])  # [1, M]
    other_area = np.clip(other_area, a_min=1e-6, a_max=None)

    iof = intersection / other_area
    return iof


def get_content(item: dict):
    try:
        if "pages" in item:
            item.pop("pages")
        if "reactions" in item:
            return json.dumps(item["reactions"])
        elif "placeholders" in item:
            if "html" in item:
                item["structure"] = item.pop("html")
            if "text" in item:
                item.pop("text")
            return json.dumps({"structure": item["structure"], "placeholders": item["placeholders"], "contents": item["contents"]})
        elif "markush" in item:
            if item["markush"]:
                return item["caption"]
            else:
                return item["smi"]
        elif "data" in item:
            return item["data"]
        elif "desc" in item:
            return item["desc"]
        elif "latex_repr" in item:
            return item["latex_repr"]
        elif "text" in item:
            return item["text"]
        else:
            return ""
    except Exception:
        return json.dumps(item)




def find_grouped_figures(original_json, out_dir="./manuscript_figures",out_name="pro_sub_figures.json"):


    if isinstance(original_json, str):
        with open(original_json, "r") as f:
            pages_dict = json.load(f)
    elif isinstance(original_json, dict):
        pages_dict = original_json
    else:
        raise ValueError("Input should be a file path or a dictionary.")

    results = []
    for page_idx, page in enumerate(pages_dict):
        result = []

        if not page:
            results.append(result)
            continue

        # print(page_idx)
        groups = []
        bboxes = []
        types = []
        contents = []
        for item in page:
            if item["type"] == "group":
                groups.append(
                    [
                        item["bbox"]["x1"],
                        item["bbox"]["y1"],
                        item["bbox"]["x2"],
                        item["bbox"]["y2"],
                    ]
                )
            else:
                bboxes.append(
                    [
                        item["bbox"]["x1"],
                        item["bbox"]["y1"],
                        item["bbox"]["x2"],
                        item["bbox"]["y2"],
                    ]
                )
                types.append(item["type"])
                contents.append(get_content(item))

        if not groups:
            results.append(result)
            continue

        if not groups or not bboxes:
            results.append(result)
            continue

        groups = np.array(groups).reshape(-1, 4).astype(np.float32)
        bboxes = np.array(bboxes).reshape(-1, 4).astype(np.float32)

        # find the bbox which iof is 1
        iof_matrix = compute_iof(groups, bboxes)  # [N, M]
        # print(iof_matrix)

        for group_idx, group_bbox in enumerate(groups):
            inside = np.where(iof_matrix[group_idx] >= 1)[0].tolist()
            # print(group_idx)
            if not inside:
                # print("no inside")
                continue

            group_bbox = group_bbox.tolist()
            inside_bboxes = bboxes[inside].tolist()
            inside_types = [types[i] for i in inside]
            inside_contents = [contents[i] for i in inside]

            # filter
            figure_related = [
                i
                for i, t in enumerate(inside_types)
                if t
                in [
                    "figure",
                    "expression",
                    "caption",
                    "paragraph",
                    "chart",
                    "legend",
                    "token",
                    "title",
                ]
            ]
            # figure_related = list(range(len(inside_types)))
            if not figure_related:
                continue

            inside_bboxes = [inside_bboxes[i] for i in figure_related]
            inside_types = [inside_types[i] for i in figure_related]
            inside_contents = [inside_contents[i] for i in figure_related]

            if all(
                [
                    "figure" not in inside_types,
                    "chart" not in inside_types,
                    "expression" not in inside_types,
                ]
            ):
                continue

            group_item = dict(
                page=page_idx,
                bbox=group_bbox,
                inside_bboxes=inside_bboxes,
                inside_types=inside_types,
                inside_contents=inside_contents,
            )
            # print(json.dumps(group_item, indent=4))
            result.append(group_item)
        results.append(result)


    # save results to json file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_filepath = os.path.join(out_dir, out_name)
    with open(out_filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_filepath}")
    return out_filepath



def get_sub_figures(sub_figure_json, pdf_file, out_dir="./manuscript_figures", out_name="sub_figures.json"):

    doi = pathlib.Path(pdf_file).stem

    doi = name_to_doi(doi)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)


    with open(sub_figure_json, "r") as f:
        sub_figures = json.load(f)

    lost_label=1

    main_res=[]
    res = []

    for dic in itertools.chain.from_iterable(sub_figures):
        page = dic["page"]
        bbox = dic["bbox"]
        inside_bboxes = dic["inside_bboxes"]
        inside_types = dic["inside_types"]
        inside_contents = dic["inside_contents"]

        caption = inside_contents[inside_types.index("caption")] if "caption" in inside_types else ""


        if match := re.compile(r'\bfig.?(?:ure)? ?([a-zA-Z])*(\d+)\b').search(caption.lower()):
            sf = match.group(1) if match.group(1) else ""
            id_num= match.group(2)
            fig_id = "figure " + sf + id_num
        else:
            continue

        if "caption" in inside_types:
            cap_bbox = inside_bboxes[inside_types.index("caption")]
            min_x = min(bbox[0], cap_bbox[0])
            min_y = min(bbox[1], cap_bbox[1])
            max_x = max(bbox[2], cap_bbox[2])
            max_y = max(bbox[3], cap_bbox[3])

            bbox = [min_x, min_y, max_x, max_y]

        main_dict = {"path":f"{fig_id}.png",
                     "caption": caption,
                     "page": page,
                     "float_xyxy": bbox,
                     "doi": doi,
                     "id": fig_id}

        extract_pdf_region_to_png(pdf_file,output_png_path=out_dir/f"{fig_id}.png",
                                  page_num=page+1,percent_rect=bbox,dpi = 300)

        main_res.append(main_dict)

        rank = 1

        for inside_bbox, inside_type, inside_content in zip(inside_bboxes, inside_types, inside_contents):

            if inside_type in ["figure", "chart"]:

                pname = f"{fig_id}({rank}).png"
                sub_dict= {"path":pname,
                          "caption": caption,
                          "page": page,
                          "float_xyxy": inside_bbox,
                          "id": fig_id,
                          "doi": doi,
                          "tag": inside_content,
                          "sub_id": rank}


                extract_pdf_region_to_png(pdf_file, output_png_path=out_dir/pname, page_num=page+1,
                                          percent_rect=inside_bbox, dpi = 300)
                rank += 1
                res.append(sub_dict)



    with open(out_dir/out_name,"w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    with open(out_dir/"figures.json","w") as f:
        json.dump(main_res, f, indent=4, ensure_ascii=False)

    return str(out_dir/out_name)



def get_sub_figures_from_pdf(file_path, inner=False, out_dir="./manuscript_figures"):
    try:
        out_filepath = trigger_file_original(file_path, inner=inner, out_dir=out_dir, out_name="sub_figures_original.json")
        if not out_filepath:
            return None
        results = find_grouped_figures(out_filepath, out_dir=out_dir, out_name="pro_sub_figures.json")
        if not results:
            return None
        get_sub_figures(results, pdf_file=file_path, out_dir=out_dir, out_name="sub_figures.json")
    except Exception as e:
        # raise e
        print(f"Error processing file {file_path}: {e}")
        return None
    return results


# if __name__ == "__main__":
#     # 101.126.82.63:4002
#     # 192.168.192.225:4002

#     pdf_file = "/root/paperextractor/soild_data_root/10_1002-adfm_201200688/manuscript_pdf/10_1002-adfm_201200688.pdf"
#     print(pdf_file)

#     results = get_sub_figures_from_pdf(file_path=pdf_file,out_dir="./sub_figures", inner=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Get figures from PDF and save as PNG.")
    parser = add_args(parser)
    args = parser.parse_args()
    run(args)


def add_args(parser):
    parser.add_argument("input_files", nargs='*', help="List of pdf files to extractor figure.")
    parser.add_argument("-pf", "--paths_file", default=None, help="File containing paths to pdf files, one per line.")
    parser.add_argument("-o", "--out_dir", default=None, help="Output directory to save extracted figures.")
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

    for pi in args.input_files:

        pi = pathlib.Path(pi)
        ppi_name = pi.parent.name
        j_name = ppi_name.replace("_pdf", "_figures")
        j_name = j_name.replace("_pdfs", "_figures")

        if args.out_dir:
            out_dir = pathlib.Path(args.out_dir) / pi.parent.parent.name
        else:
            out_dir = pi.parent.parent
        out_dir = out_dir / j_name

        get_sub_figures_from_pdf(pi, out_dir=out_dir)
        print(f"Processed {pi} and saved results to {out_dir}",flush=True)


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     pi = "/root/paperextractor/soild_data_root/10_1016-j_memsci_2013_07_043/manuscript_pdf/10_1016-j_memsci_2013_07_043.pdf"
#     pi = pathlib.Path(pi)
#     ppi_name = pi.parent.name
#     j_name = ppi_name.replace("_pdf", "_figures")
#     out_dir = pi.parent.parent / j_name

#     get_sub_figures_from_pdf(pi, out_dir=out_dir)