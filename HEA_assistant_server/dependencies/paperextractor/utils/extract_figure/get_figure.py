import json
import os
import pathlib
import re
from typing import Tuple

import fitz

from paperextractor.utils.doi_to_name import name_to_doi


def extract_pdf_region_to_png(
    pdf_path: str,
    output_png_path: str,
    page_num: int,
    percent_rect: Tuple[float, float, float, float],
    dpi: int = 300
) -> None:
    """
    Extract a specific region from a PDF page using percentage coordinates and save as PNG.

    Args:
        pdf_path (str): Path to the input PDF file
        output_png_path (str): Path where the output PNG file will be saved
        page_num (int): Page number to extract from (1-based indexing)
        percent_rect (tuple): Region coordinates as percentages (left, top, right, bottom)

    Note:
        The percent_rect coordinates should be provided as decimal values between 0 and 1
        representing the percentage of the page dimensions.

    Raises:
        FileNotFoundError: If the input PDF file does not exist
        ValueError: If page_num is less than 1 or if percent_rect values are invalid
        RuntimeError: If the PDF file is corrupted or cannot be read
    """
    # Validate input PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")

    # Validate page number
    if page_num < 1:
        raise ValueError("Page number must be greater than 0")

    # Validate percent_rect values
    for value in percent_rect:
        if not 0 <= value <= 1:
            raise ValueError("Percent coordinates must be between 0 and 1")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_png_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Open the PDF document
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF file: {str(e)}")

    try:
        # Validate PDF is readable
        if not doc.is_pdf:
            raise RuntimeError("The file is not a valid PDF document")

        # Validate PDF is not encrypted
        if doc.is_encrypted:
            raise RuntimeError("The PDF file is encrypted and cannot be read")

        # Validate page number is within document bounds
        if page_num > len(doc):
            raise ValueError(f"Page number {page_num} exceeds document length of {len(doc)}")

        # Get the specified page (convert from 1-based to 0-based indexing)
        page = doc[page_num - 1]

        # Get the actual page dimensions
        page_width = page.rect.width
        page_height = page.rect.height

        # Convert percentage coordinates to actual pixel coordinates
        left = percent_rect[0] * page_width
        top = percent_rect[1] * page_height
        right = percent_rect[2] * page_width
        bottom = percent_rect[3] * page_height
        clip_rect = fitz.Rect(left, top, right, bottom)

        # Render the page and crop to the specified region
        # Using 300 DPI for high-quality output
        pix = page.get_pixmap(clip=clip_rect, dpi=dpi)

        # Save the cropped region as PNG
        pix.save(output_png_path)

    finally:
        # Clean up: close the document
        doc.close()

    return output_png_path


def get_fig_from_pdf(pdf_file=None, json_file=None, out_dir="figure") -> int:


    data = json.load(open(json_file, "r"))["objects"]
    output_path = out_dir
    # if os.path.exists(output_path):
    #     print(f"{output_path} already exists, skip.")
    #     return 0
    doi = pathlib.Path(pdf_file).stem
    doi = name_to_doi(doi)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    figures = []
    caption = []
    for ii in data:
        if ii["class"] == "chart":
            figures.append(ii)
        elif ii["class"] == "caption":
            caption.append(ii)

    figures_info = {}

    for index, ii in enumerate(figures):
        page = ii["page"]
        bottom = ii["float_xyxy"][3]
        distance = 1
        possible_caption = {"class": "caption",
                            "confidence": 0,
                            "position": [0, 0, 0, 0],
                            "page_num":0 ,
                            "text": "No caption found.",
                            "id": "" }
        # find the nearest caption
        for jj in caption:
            if jj["page"] != page:
                continue
            if jj["float_xyxy"][1] < bottom:
                continue
            if jj["float_xyxy"][1] - bottom < distance:
                distance = jj["float_xyxy"][1] - bottom
                possible_caption = jj

        # if caption too short, try find the next caption
        other_caption = caption.copy()
        while  len(possible_caption["str"]) < 50:
            print(possible_caption)
            distance = 1
            try:
                other_caption.remove(possible_caption)
            except:
                print("No other caption found.")
                break
            for jj in other_caption:
                if jj["page"] != page:
                    continue
                if jj["float_xyxy"][1] < bottom:
                    continue
                if jj["float_xyxy"][1] - bottom < distance:
                    distance = jj["float_xyxy"][1] - bottom
                    possible_caption = jj

        # if no caption found, try find upper caption
        if distance == 1:
            for jj in caption:
                if jj["page"] != page:
                    continue
                if jj["float_xyxy"][1] > bottom - 0.05:
                    continue
                if bottom - jj["float_xyxy"][1] < distance:
                    distance = bottom - jj["float_xyxy"][1]
                    possible_caption = jj

        # # center of figure usually align with caption.
        # center_figure = (ii["position"][0] + ii["position"][2]) / 2
        # center_caption = (possible_caption["position"][0] + possible_caption["position"][2]) / 2
        # if abs(center_figure - center_caption) > 0.02:
        #     print(f"Page {page+1} has a figure and caption not align, "
        #           "maybe it is part of figure panel.")
        # # distance between caption and caption usuallly not exceed 2 %.
        # elif possible_caption["position"][1] - bottom > 0.02:
        #     print(f"Page {page+1} has a figure and caption too far, "
        #           "maybe it is part of figure panel.")

        figures_info[f"{index}"] = possible_caption

    # deduplicate the caption, find the largest figure + caption
    # should union all figures with same caption.
    unique_figures_info = []
    unique_captions = []
    for ii in figures_info.values():
        if ii not in unique_captions:
            unique_captions.append(ii)
    print("Unique captions: ", len(unique_captions))

    for index1, ii in enumerate(unique_captions):
        area = 0
        left = ii["float_xyxy"][0]
        top = ii["float_xyxy"][1]
        right = ii["float_xyxy"][2]
        bottom = ii["float_xyxy"][3]
        for index2, jj in enumerate(figures):
            if figures_info[f"{index2}"] == ii:
                left = min(jj["float_xyxy"][0], left)
                top = min(jj["float_xyxy"][1], top)
                right = max(jj["float_xyxy"][2], right)
                bottom = max(jj["float_xyxy"][3], bottom)
                if (right - left) * (bottom - top) > area:
                    area = (right - left) * (bottom - top)
                    bounding_box = [left, top, right, bottom]

        unique_figure_json = {}
        page = ii["page"]
        unique_figure_json["doi"] = doi
        unique_figure_json["path"] = f"{index1}.png"
        unique_figure_json["caption"] = ii["str"]
        unique_figure_json["page"] = page + 1
        unique_figure_json["float_xyxy"] = bounding_box
        # try find the figure with id, such as 'fig 1', 'figure 1', etc. use re to find the number.
        pattern = r'\b(fig.?(?:ure)? ?\s*\d+)\b'
        matches = re.findall(pattern, ii["str"], flags=re.IGNORECASE)
        if len(matches) > 0:
            unique_figure_json["id"] = matches[0]
        else:
            print(ii["str"])
            unique_figure_json["id"] = f""
            print(f"Page {page+1} has a figure caption without 'fig', "
                "maybe it is not a figure caption.\n")

        # crop the figure
        extract_pdf_region_to_png(pdf_file, f"{output_path}/{index1}.png", page+1,
                                  bounding_box)

        unique_figures_info.append(unique_figure_json)

    with open(f"{output_path}/figures.json", "w") as f:
        json.dump(unique_figures_info, f, indent=4)

def get_figures_from_pdf(pdf_file: str, json_file: str, out_dir,
                         sub_dir="manuscript_figures"):

    out_dir = os.path.join(out_dir, sub_dir)
    try:
        get_fig_from_pdf(
        pdf_file=pdf_file,
        json_file=json_file,
        out_dir=out_dir
        )
    except Exception as e:
        print(f"Error processing figure {pdf_file}: {e}")
        return None

# if __name__ == "__main__":
#     get_fig_from_pdf(
#         pdf_file="/root/paperextractor/soild_data_root/10_1002-adfm_201200688/manuscript_pdf/10_1002-adfm_201200688.pdf",
#         json_file="/root/paperextractor/soild_data_root/10_1002-adfm_201200688/manuscript_sparse_json/10_1002-adfm_201200688.json",
#         out_dir="/root/paperextractor/soild_data_root/10_1002-adfm_201200688/manuscript_figures"
#     )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Get figures from PDF and save as PNG.")
    parser = add_args(parser)
    args = parser.parse_args()
    run(args)


def add_args(parser):
    parser.add_argument("input_files", nargs='*', help="List of pdf files to extractor figure.")
    parser.add_argument("-pf", "--paths_file", default=None, help="File containing paths to pdf files, one per line.")
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
        file_stem = pi.stem
        pppi_name = pi.parent.parent.name
        ppi_name = pi.parent.name
        j_name = ppi_name.replace("_pdf", "_sparse_json")
        json_dir = pi.parent / f"{j_name}"
        json_file = pi.parent / f"{j_name}/ {file_stem}.json"
        if not json_file.is_file():
            from llmchat.sparse_pdf import read_pdf_txt_uni_sparser
            w, response = read_pdf_txt_uni_sparser(str(pi))
            txt_dir = ppi_name.replace("_pdf", "_sparse_txt")

            os.makedirs(txt_dir, exist_ok=True)
            with open(os.path.join(txt_dir, f"{file_stem}.txt"), "w") as f:
                f.write(w)


            os.makedirs(json_dir, exist_ok=True)
            with open(os.path.join(json_dir, f"{file_stem}.json"), "w") as f:
                json.dump(response, f, indent=4, ensure_ascii=False)

        get_figures_from_pdf(pi, json_file, str(pppi_name),
                         sub_dir="manuscript_figures")


if __name__ == "__main__":
    main()