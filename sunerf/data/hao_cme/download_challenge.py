#!/usr/bin/env python3
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Download CME Challenge FITS files (stepnum_005.fits to stepnum_099.fits) from multiple directories."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory where files will be saved."
    )
    args = parser.parse_args()

    # base_urls = [
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_090E_bang_0000_tB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_090E_bang_0000_pB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_030E_bang_0000_tB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_030E_bang_0000_pB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_060W_bang_0000_tB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_060W_bang_0000_pB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_0000_bang_0000_tB",
    #     f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme1/cme1_dcmer_0000_bang_0000_pB",
    # ]
    base_urls = [
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_090E_bang_0000_tB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_090E_bang_0000_pB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_030E_bang_0000_tB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_030E_bang_0000_pB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_060W_bang_0000_tB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_060W_bang_0000_pB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_0000_bang_0000_tB",
        f"https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/cme2/cme2_dcmer_0000_bang_0000_pB",
    ]

    os.makedirs(args.output_path, exist_ok=True)

    for base_url in base_urls:
        subdir_name = base_url.rstrip("/").split("/")[-1]
        subdir_path = os.path.join(args.output_path, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)

        print(f"\n=== Downloading from {subdir_name} ===")

        for i in range(5, 100):
            filename = f"stepnum_{i:03d}.fits"
            filepath = os.path.join(subdir_path, filename)
            if os.path.exists(filepath):
                print(f"Skipping {filename} (already exists)")
                continue

            url = f"{base_url}/{filename}"
            print(f"Downloading {filename} ...")
            subprocess.run(["wget", "-q", "-P", subdir_path, url], check=False)

    print(f"\nAll downloads completed. Files saved in: {args.output_path}")

if __name__ == "__main__":
    main()
