import argparse
from segmentation import segment_nii_files
from rendering import generate_3d_rendering

def main():
    parser = argparse.ArgumentParser(description="NIfTI (.nii) File Segmentation Tool")
    parser.add_argument("input_directory", help="Directory with .nii files")
    parser.add_argument("class_name", help="Class name for segmentation (e.g., spinal_cord)")
    parser.add_argument("model_id", help="Model id for the class, format will be NCA_(NUM_NCA)_(underscore seperated input size)")
    parser.add_argument("--output_directory", help="Output directory for saving segmentation masks")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Choose CPU or GPU")
    parser.add_argument("--preview", action="store_true", help="Generate a 3D rendering preview")

    args = parser.parse_args()

    if args.preview:
        generate_3d_rendering(args.input_directory, args.class_name, args.output_directory, args.device, args.model_id)
    else:
        segment_nii_files(args.input_directory, args.class_name, args.output_directory, args.device)

if __name__ == "__main__":
    main()
