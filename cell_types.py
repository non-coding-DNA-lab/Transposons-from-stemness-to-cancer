#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import os
import gzip
import click
import shutil
import concurrent.futures
from Bio import SeqIO


def create_cell_type_dict(df, level, score_threshold, assay):
    cells = {}
    filtered_df = df[df[f"predicted.ann_level_{level}.score"] > score_threshold]
    for cell_type in filtered_df[f"predicted.ann_level_{level}"].unique():
        cells.update(
            filtered_df[
                (filtered_df[f"predicted.ann_level_{level}"] == cell_type)
                & (filtered_df.assay_name == assay)
            ]
            .set_index("cell")[f"predicted.ann_level_{level}"]
            .to_dict()
        )
    return cells


def divide_fastq_by_cell_type(
    fastq_file_R1_gz, fastq_file_R2_gz, cell_types, assay, output_dir, index_len=16
):
    """
    Divides gzipped FASTQ files (R2) into groups based on cell types extracted from another gzipped
    FASTQ file (R1).
    """
    # Check if output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to hold file handles for each cell type
    file_handles = {}

    cell_barcode_to_type_level1 = create_cell_type_dict(
        cell_types.copy(), 1, 0.9, assay
    )
    cell_barcode_to_type_level2 = create_cell_type_dict(
        cell_types.copy(), 2, 0.8, assay
    )
    cell_barcode_to_type_level3 = create_cell_type_dict(
        cell_types.copy(), 3, 0.7, assay
    )

    # Read the gzipped FASTQ files
    with gzip.open(fastq_file_R1_gz, "rt") as handle_R1, gzip.open(
        fastq_file_R2_gz, "rt"
    ) as handle_R2:
        for record_R1, record_R2 in zip(
            SeqIO.parse(handle_R1, "fastq"), SeqIO.parse(handle_R2, "fastq")
        ):
            # Extract cell barcode from the R1 record (first 16 nucleotides)
            cell_barcode = str(record_R1.seq)[:index_len]

            # Find the corresponding cell type
            cell_type = cell_barcode_to_type_level1.get(cell_barcode)

            if cell_type:
                # Check if file handle exists, create if not
                if cell_type not in file_handles:
                    file_handles[cell_type] = gzip.open(
                        os.path.join(output_dir, f"Level1_{cell_type}.fastq.gz"), "wt"
                    )

                # Write the R2 record to the corresponding file
                SeqIO.write(record_R2, file_handles[cell_type], "fastq")

            cell_type = cell_barcode_to_type_level2.get(cell_barcode)

            if cell_type:
                # Check if file handle exists, create if not
                if cell_type not in file_handles:
                    file_handles[cell_type] = gzip.open(
                        os.path.join(output_dir, f"Level2_{cell_type}.fastq.gz"), "wt"
                    )

                # Write the R2 record to the corresponding file
                SeqIO.write(record_R2, file_handles[cell_type], "fastq")

            cell_type = cell_barcode_to_type_level3.get(cell_barcode)

            if cell_type:
                # Check if file handle exists, create if not
                if cell_type not in file_handles:
                    file_handles[cell_type] = gzip.open(
                        os.path.join(output_dir, f"Level3_{cell_type}.fastq.gz"), "wt"
                    )

                # Write the R2 record to the corresponding file
                SeqIO.write(record_R2, file_handles[cell_type], "fastq")

    # Close all file handles
    for handle in file_handles.values():
        handle.close()

    print(f"FASTQ file - {assay}, divided by cell types. Output in {output_dir}")


def merge_files_by_cell_type(base_output_dir, final_output_dir):
    """
    Merge files by cell type into a single file per cell type.
    """
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    cell_type_files = {}
    # Walk through the base output directory to find all files
    for root, dirs, files in os.walk(base_output_dir):
        for file in files:
            if file.endswith(".fastq.gz"):
                cell_type = file.replace(".fastq.gz", "")
                if cell_type not in cell_type_files:
                    cell_type_files[cell_type] = []
                cell_type_files[cell_type].append(os.path.join(root, file))

    # Merge files for each cell type
    for cell_type, file_paths in cell_type_files.items():
        with gzip.open(
            os.path.join(final_output_dir, f"{cell_type}.fastq.gz"), "wt"
        ) as output_file:
            for file_path in file_paths:
                with gzip.open(file_path, "rt") as input_file:
                    shutil.copyfileobj(input_file, output_file)

    # Optionally, clean up the individual output directories
    shutil.rmtree(base_output_dir)


def parallel_process_fastq_files(
    fastq_pairs, cell_types, output_dir, index_len, max_workers=4
):
    """
    Process multiple pairs of FASTQ files in parallel, each pair writing to a unique output directory.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (assay, R1, R2) in enumerate(fastq_pairs):
            # Create a unique output directory for each pair
            output_dir_pair = os.path.join(output_dir, f"pair_{assay}")
            os.makedirs(output_dir_pair, exist_ok=True)

            # Submit each pair of FASTQ files to the executor
            futures.append(
                executor.submit(
                    divide_fastq_by_cell_type,
                    R1,
                    R2,
                    cell_types,
                    assay,
                    output_dir_pair,
                    index_len,
                )
            )

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing files: {e}")

    final_output_dir = os.path.join(output_dir, "final_merged_files")
    merge_files_by_cell_type(output_dir, final_output_dir)


@click.command()
@click.option(
    "--fastq_dir", type=click.Path(), help="fastq files directory", required=True
)
@click.option("--output_dir", type=click.Path(), help="output directory", required=True)
@click.option("--azimuth", type=click.Path(), help="azimuth file path", required=True)
@click.option("-t", type=click.INT, default=4, required=True)
def main(fastq_dir, output_dir, azimuth, t):
    cell_types = pd.read_csv(
        azimuth,
        sep="\t",
        names=[
            "cell",
            "predicted.ann_level_3",
            "predicted.ann_level_2",
            "predicted.ann_level_1",
            "predicted.ann_level_3.score",
            "predicted.ann_level_2.score",
            "predicted.ann_level_1.score",
            "mapping.score",
        ],
    )

    cell_types["assay_name"] = cell_types["cell"].apply(lambda x: x.split(":")[1])
    cell_types["cell"] = cell_types["cell"].apply(lambda x: x.split(":")[0])

    assay_name = [
        "BT1290",
        "BT1291",
        "BT1292",
        "BT1293",
        "BT1294",
        "BT1295",
        "BT1296",
        "BT1297",
        "BT1298",
        "BT1299",
        "BT1300",
        "BT1301",
        "scrBT1430m",
        "scrBT1431m",
        "scrBT1432m",
        "scrBT1425",
        "scrBT1426",
        "scrBT1427",
        "scrBT1428",
        "scrBT1429m",
        "BT1375",
        "BT1376",
        "BT1377",
        "BT1378",
    ]
    fastq_pairs = []
    for assay in assay_name:
        file1 = f"{fastq_dir}/{assay}_S1_L001_R1_001.fastq.gz"
        file2 = f"{fastq_dir}/{assay}_S1_L001_R2_001.fastq.gz"
        fastq_pairs.append((assay, file1, file2))

    parallel_process_fastq_files(fastq_pairs, cell_types, output_dir, 16, t)

    assay_name = ["1247", "BT1A", "BT1249", "BT1C", "BT2B", "BT1B", "BT2A"]
    fastq_pairs = []
    for assay in assay_name:
        file1 = f"{fastq_dir}/{assay}_S1_L001_I1_001.fastq.gz"
        file2 = f"{fastq_dir}/{assay}_S1_L001_R1_001.fastq.gz"
        fastq_pairs.append((assay, file1, file2))

    parallel_process_fastq_files(fastq_pairs, cell_types, output_dir, 14, t)


if __name__ == "__main__":
    main()
