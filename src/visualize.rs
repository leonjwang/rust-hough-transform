use image::Rgba;
use imageproc::drawing::draw_line_segment_mut;

// fn dump_houghspace(accumulator: &na::DMatrix<u32>, houghspace_img_path: PathBuf) {
//     let max_accumulator_value = matrix_max(accumulator).unwrap_or(0);

//     println!("# max accumulator value: {}", max_accumulator_value);

//     let out_img_width = accumulator.nrows() as u32;
//     let out_img_height = accumulator.ncols() as u32;

//     let mut out = ImageBuffer::new(out_img_width, out_img_height);

//     for y in 0..out_img_height {
//         for x in 0..out_img_width {
//             let n = std::cmp::min(
//                 ((accumulator[(x as usize, y as usize)] as f64) * 255.0
//                     / (max_accumulator_value as f64))
//                     .round() as u32,
//                 255,
//             ) as u8;

//             let pixel = Rgb([n, n, n]);

//             out[(x, out_img_height - y - 1)] = pixel;
//         }
//     }

//     // Modern image crate handles file creation and format deduction natively
//     out.save(houghspace_img_path)
//         .expect("Failed to save houghspace image");
// }

use std::path::PathBuf;

// use crate::arithmetic::deg2rad;

const LINE_COLOR: Rgba<u8> = Rgba([255, 0, 0, 255]);

use image::GenericImageView;

pub fn dump_line_visualization(
    img: &mut image::DynamicImage,
    lines_rho_theta: &[(f64, f64)], // idiomatic slice instead of &Vec
    line_visualization_img_path: PathBuf,
) {
    let (img_width, img_height) = img.dimensions();

    let mut lines = vec![];

    // Notice we no longer need the theta_axis_scale_factor!
    for &(rho, theta_rad) in lines_rho_theta {
        let line_coordinates = line_from_rho_theta(rho, theta_rad, img_width, img_height);

        lines.push(line_coordinates);
    }

    println!("# detected lines: {}", lines.len());

    for line_coordinates in lines {
        let res = clip_line_liang_barsky(
            (0, (img_width - 1) as i32, 0, (img_height - 1) as i32),
            line_coordinates,
        );

        if let Some(clipped_line_coordinates) = res {
            // Define LINE_COLOR here, or ensure it's in scope.
            // e.g., let LINE_COLOR = image::Rgba([255u8, 0u8, 0u8, 255u8]);

            draw_line_segment_mut(
                img,
                // be sure to not overflow height
                (
                    clipped_line_coordinates.0 as f32,
                    (img_height as f32) - 1.0 - clipped_line_coordinates.1 as f32,
                ),
                (
                    clipped_line_coordinates.2 as f32,
                    (img_height as f32) - 1.0 - clipped_line_coordinates.3 as f32,
                ),
                LINE_COLOR, // Make sure this constant exists in your file
            );
        }
    }

    img.save(line_visualization_img_path)
        .expect("Failed to save line visualization image");
}

fn line_from_rho_theta(
    rho: f64,
    theta_rad: f64,
    img_width: u32,
    img_height: u32,
) -> (i32, i32, i32, i32) {
    let a = theta_rad.cos();
    let b = theta_rad.sin();

    // (x0, y0) is the point on the line closest to the origin
    let x0 = a * rho;
    let y0 = b * rho;

    // A length guaranteed to span completely across the image
    let length = ((img_width as f64).hypot(img_height as f64)) * 2.0;

    // Extend outward from (x0, y0) in both directions using the tangent vector (-b, a)
    let p1_x = x0 + length * (-b);
    let p1_y = y0 + length * (a);

    let p2_x = x0 - length * (-b);
    let p2_y = y0 - length * (a);

    (
        p1_x.round() as i32,
        p1_y.round() as i32,
        p2_x.round() as i32,
        p2_y.round() as i32,
    )
}

#[allow(unused_assignments)]
fn clip_line_liang_barsky(
    clipping_area: (i32, i32, i32, i32), /* Define the x/y clipping values for the border. */
    line_coordinates: (i32, i32, i32, i32),
) -> Option<(i32, i32, i32, i32)> {
    let (edge_left, edge_right, edge_bottom, edge_top) = clipping_area;
    let (x0src, y0src, x1src, y1src) = line_coordinates;

    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;

    let xdelta = (x1src as f64) - (x0src as f64);
    let ydelta = (y1src as f64) - (y0src as f64);

    let mut p = 0.0f64;
    let mut q = 0.0f64;
    let mut r = 0.0f64;

    for edge in 0..4 {
        // Traverse through left, right, bottom, top edges.
        if edge == 0 {
            p = -xdelta;
            q = -((edge_left as f64) - (x0src as f64));
        }
        if edge == 1 {
            p = xdelta;
            q = (edge_right as f64) - (x0src as f64);
        }
        if edge == 2 {
            p = -ydelta;
            q = -((edge_bottom as f64) - (y0src as f64));
        }
        if edge == 3 {
            p = ydelta;
            q = (edge_top as f64) - (y0src as f64);
        }
        r = q / p;

        if p == 0.0 && q < 0.0 {
            // Don't draw line at all. (parallel line outside)
            return None;
        }

        if p < 0.0 {
            if r > t1 {
                return None;
            } else if r > t0 {
                t0 = r;
            }
        } else if p > 0.0 {
            if r < t0 {
                return None;
            } else if r < t1 {
                t1 = r;
            }
        }
    }

    let x0clip = (x0src as f64) + t0 * (xdelta as f64);
    let y0clip = (y0src as f64) + t0 * (ydelta as f64);
    let x1clip = (x0src as f64) + t1 * (xdelta as f64);
    let y1clip = (y0src as f64) + t1 * (ydelta as f64);

    Some((
        x0clip.round() as i32,
        y0clip.round() as i32,
        x1clip.round() as i32,
        y1clip.round() as i32,
    ))
}
