use std::f64::consts::PI;

use image::DynamicImage;
use imageproc::drawing::Canvas;

#[inline]
pub fn deg2rad(deg: u32, axis_size: u32) -> f64 {
    // compute radians based on the theta axis size, which can be greater than 180 deg
    let pi: f64 = std::f64::consts::PI;
    deg as f64 * (pi / axis_size as f64)
}

#[inline]
pub fn calculate_max_line_length(img_width: u32, img_height: u32) -> f64 {
    ((img_width as f64).hypot(img_height as f64)).ceil()
}

#[inline]
fn rgb_to_greyscale(r: u8, g: u8, b: u8) -> u8 {
    ((r as f64 + g as f64 + b as f64) / 3.0).round() as u8
}

#[inline]
fn is_edge(pixel: &image::Rgba<u8>) -> bool {
    // channels() is deprecated. We can index directly into the Rgba array.
    let greyscale_value = rgb_to_greyscale(pixel[0], pixel[1], pixel[2]);
    greyscale_value < 1
}

#[inline]
fn scale_rho(rho: f64, rho_axis_size: u32, max_line_length: f64) -> u32 {
    let rho_axis_half = (rho_axis_size as f64 / 2.0).round();
    ((rho * rho_axis_half / max_line_length).round() + rho_axis_half) as u32
}

#[inline]
fn invert_y(img_height: u32, coords: &(u32, u32)) -> (u32, u32) {
    let &(x, y) = coords;
    let y_inverted = img_height - y - 1;

    (x, y_inverted)
}

#[inline]
fn calculate_rho(theta: u32, theta_axis_size: u32, coords: &(u32, u32)) -> f64 {
    let &(x, y) = coords;
    let sin = deg2rad(theta, theta_axis_size).sin();
    let cos = deg2rad(theta, theta_axis_size).cos();

    (x as f64) * cos + (y as f64) * sin
}

pub fn hough_transform(
    img: &image::DynamicImage,
    theta_axis_scale_factor: u32,
    rho_axis_scale_factor: u32,
) -> nalgebra::DMatrix<u32> {
    let max_line_length = calculate_max_line_length(img.width(), img.height());
    let theta_axis_size = theta_axis_scale_factor * 180;
    let rho_axis_size = (max_line_length as u32) * rho_axis_scale_factor;

    let mut pixel_coords = (0..img.width())
        .flat_map(|x| (0..img.height()).map(move |y| (x, y)))
        .collect::<Vec<_>>();

    pixel_coords.sort_by_key(|&(x, y)| (!y, x));

    pixel_coords
        .iter()
        .filter(|&&(x, y)| is_edge(&img.get_pixel(x, y)))
        .map(|&coords| invert_y(img.height(), &coords))
        .flat_map(|coords| {
            (0..theta_axis_size)
                .map(move |theta| (theta, calculate_rho(theta, theta_axis_size, &coords)))
        })
        .fold(
            nalgebra::DMatrix::from_element(theta_axis_size as usize, rho_axis_size as usize, 0),
            |mut accu, (theta, rho)| {
                let rho_scaled = scale_rho(rho, rho_axis_size, max_line_length);
                accu[(theta as usize, rho_scaled as usize)] += 1;
                accu
            },
        )
}

pub fn get_lines(
    img: &DynamicImage,
    accumulator: &nalgebra::DMatrix<u32>,
    houghspace_filter_threshold: u32,
) -> Vec<(f64, f64)> {
    let (img_width, img_height) = img.dimensions();

    let theta_axis_size = accumulator.nrows();
    let rho_axis_size = accumulator.ncols();
    let rho_axis_half = ((rho_axis_size as f64) / 2.0).round();
    let max_line_length = calculate_max_line_length(img_width, img_height);

    let mut lines_rho_theta = vec![];

    for theta in 0..theta_axis_size {
        for rho_scaled in 0..rho_axis_size {
            let val = accumulator[(theta as usize, rho_scaled as usize)];

            if val < houghspace_filter_threshold {
                continue;
            }

            let rho = (rho_scaled as f64 - rho_axis_half) * max_line_length / rho_axis_half;

            let theta = (theta as f64) * (PI / theta_axis_size as f64);

            lines_rho_theta.push((rho, theta as f64));
        }
    }

    lines_rho_theta
}

// Thank you gemini
pub fn estimate_center(lines: &Vec<(f64, f64)>) -> (f64, f64, f64) {
    let n = lines.len();

    let mut a_matrix = nalgebra::DMatrix::<f64>::zeros(n, 3);
    let mut b_vector = nalgebra::DVector::<f64>::zeros(n);

    for (i, &(rho, theta)) in lines.iter().enumerate() {
        a_matrix[(i, 0)] = theta.cos();
        a_matrix[(i, 1)] = theta.sin();
        a_matrix[(i, 2)] = 1.0; // The constant C (representing radius)
        b_vector[i] = rho;
    }

    // Solve Ax = B using SVD for Least Squares
    // This finds the best fit x_c, y_c, and C across all your tangent lines
    let svd = nalgebra::SVD::new(a_matrix, true, true);
    let solution = svd
        .solve(&b_vector, 1e-6)
        .expect("Failed to solve the linear system");

    let x_center = solution[0];
    let y_center = solution[1];

    // The radius is the absolute value of the 3rd parameter
    let radius = solution[2].abs();

    (x_center, y_center, radius)
}
