extern crate image;
extern crate imageproc;
extern crate nalgebra;

use std::str::FromStr;
use std::{env, path::PathBuf, process};

use hough_transform::detect_center;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();

    if args.len() < 3 {
        eprintln!(
            "Usage: hough-image input_path output_path theta_axis_scale_factor rho_axis_scale_factor houghspace_filter_threshold"
        );
        process::exit(1);
    }

    let input_img_path = PathBuf::from(args[0].to_string());

    let theta_axis_scale_factor =
        u32::from_str(&args[2]).expect("ERROR 'theta_axis_scale_factor' argument not a number.");
    let rho_axis_scale_factor =
        u32::from_str(&args[3]).expect("ERROR 'rho_axis_scale_factor' argument not a number.");
    let houghspace_filter_offset = u32::from_str(&args[4])
        .expect("ERROR 'houghspace_filter_threshold' argument not a number.");

    detect_center(
        input_img_path,
        theta_axis_scale_factor,
        rho_axis_scale_factor,
        houghspace_filter_offset,
    );
}
