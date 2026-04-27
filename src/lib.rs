use std::path::PathBuf;

use image::DynamicImage;

use crate::visualize::dump_line_visualization;

mod arithmetic;
mod preprocess;
mod visualize;

fn matrix_max<T: Copy + std::cmp::Ord>(matrix: &nalgebra::DMatrix<T>) -> Option<T> {
    matrix.iter().max().copied()
}

pub fn detect_center(
    input_path: PathBuf,
    theta_axis_scale_factor: u32,
    rho_axis_scale_factor: u32,
    houghspace_filter_offset: u32,
) {
    let src = image::open(&input_path).expect("ERROR: input file not found.");
    let mut shapes: Vec<DynamicImage> = preprocess::get_shapes(src);

    for (i, img) in shapes.iter_mut().enumerate() {
        img.save(PathBuf::from("out").join(format!(
            "{0}-{1}.png",
            input_path.file_name().unwrap().to_string_lossy(),
            i
        )))
        .unwrap();

        let accu =
            arithmetic::hough_transform(&img, theta_axis_scale_factor, rho_axis_scale_factor);

        // dump_houghspace(
        //     &accu,
        //     PathBuf::from("out").join(format!(
        //         "{0}-{1}-shape.png",
        //         input_path.file_name().unwrap().to_string_lossy(),
        //         i
        //     )),
        // );

        let max_accumulator_value = matrix_max(&accu).unwrap_or(0);

        dump_line_visualization(
            img,
            &accu,
            theta_axis_scale_factor,
            (max_accumulator_value as i32 - houghspace_filter_offset as i32).max(0) as u32,
            PathBuf::from("out").join(format!(
                "{0}-{1}-lines.png",
                input_path.file_name().unwrap().to_string_lossy(),
                i
            )),
        );
    }
}
