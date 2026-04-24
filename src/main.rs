extern crate image;
extern crate imageproc;
extern crate nalgebra;

use std::env;
use std::f64;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

use image::DynamicImage;
use image::GenericImage;
use image::{GenericImageView, ImageBuffer, Rgb, Rgba};
use imageproc::drawing::draw_line_segment_mut;
use nalgebra as na;

const DETECT_COLOR: Rgba<u8> = Rgba([180, 20, 40, 255]); // TODO: Dynamicly detect this from image
const THRESHOLD: f64 = 50.0;

const MIN_PIXEL_AMOUNT: f64 = 0.05; // Amount of canvas pixels registered to be detected as an image

const LINE_COLOR: Rgba<u8> = Rgba([255, 0, 0, 255]);
const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);
const BLACK: Rgba<u8> = Rgba([0, 0, 0, 255]);

fn color_dist(c1: &Rgba<u8>, c2: &Rgba<u8>) -> f64 {
    return ((c1[0] as f64 - c2[0] as f64).powi(2)
        + (c1[1] as f64 - c2[1] as f64).powi(2)
        + (c1[2] as f64 - c2[2] as f64).powi(2))
    .sqrt();
}

#[inline]
fn deg2rad(deg: u32, axis_size: u32) -> f64 {
    // compute radians based on the theta axis size, which can be greater than 180 deg
    let pi: f64 = std::f64::consts::PI;
    deg as f64 * (pi / axis_size as f64)
}

#[inline]
fn calculate_max_line_length(img_width: u32, img_height: u32) -> f64 {
    ((img_width as f64).hypot(img_height as f64)).ceil()
}

#[inline]
fn rgb_to_greyscale(r: u8, g: u8, b: u8) -> u8 {
    ((r as f64 + g as f64 + b as f64) / 3.0).round() as u8
}

fn matrix_max<T: Copy + std::cmp::Ord>(matrix: &na::DMatrix<T>) -> Option<T> {
    matrix.iter().max().copied()
}

fn dump_houghspace(accumulator: &na::DMatrix<u32>, houghspace_img_path: PathBuf) {
    let max_accumulator_value = matrix_max(accumulator).unwrap_or(0);

    println!("# max accumulator value: {}", max_accumulator_value);

    let out_img_width = accumulator.nrows() as u32;
    let out_img_height = accumulator.ncols() as u32;

    let mut out = ImageBuffer::new(out_img_width, out_img_height);

    for y in 0..out_img_height {
        for x in 0..out_img_width {
            let n = std::cmp::min(
                ((accumulator[(x as usize, y as usize)] as f64) * 255.0
                    / (max_accumulator_value as f64))
                    .round() as u32,
                255,
            ) as u8;

            let pixel = Rgb([n, n, n]);

            out[(x, out_img_height - y - 1)] = pixel;
        }
    }

    // Modern image crate handles file creation and format deduction natively
    out.save(houghspace_img_path)
        .expect("Failed to save houghspace image");
}

fn dump_line_visualization(
    img: &mut image::DynamicImage,
    accumulator: &na::DMatrix<u32>,
    theta_axis_scale_factor: u32,
    houghspace_filter_threshold: u32,
    line_visualization_img_path: PathBuf,
) {
    let (img_width, img_height) = img.dimensions();

    let theta_axis_size = accumulator.nrows();
    let rho_axis_size = accumulator.ncols();
    let rho_axis_half = ((rho_axis_size as f64) / 2.0).round();
    let max_line_length = calculate_max_line_length(img_width, img_height);

    let mut lines = vec![];

    for theta in 0..theta_axis_size {
        for rho_scaled in 0..rho_axis_size {
            let val = accumulator[(theta as usize, rho_scaled as usize)];

            if val < houghspace_filter_threshold {
                continue;
            }

            let rho = (rho_scaled as f64 - rho_axis_half) * max_line_length / rho_axis_half;

            let line_coordinates = line_from_rho_theta(
                theta as u32,
                theta_axis_scale_factor,
                rho as f64,
                img_width,
                img_height,
            );

            lines.push((theta, line_coordinates));
        }
    }

    println!("# detected lines: {}", lines.len());

    for (_, line_coordinates) in lines {
        let res = clip_line_liang_barsky(
            (0, (img_width - 1) as i32, 0, (img_height - 1) as i32),
            line_coordinates,
        );

        if let Some(clipped_line_coordinates) = res {
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
                LINE_COLOR,
            );
        }
    }

    img.save(line_visualization_img_path)
        .expect("Failed to save line visualization image");
}

fn line_from_rho_theta(
    theta: u32,
    theta_axis_scale_factor: u32,
    rho: f64,
    img_width: u32,
    img_height: u32,
) -> (i32, i32, i32, i32) {
    let mut p1_x = 0.0_f64;
    let mut p1_y = 0.0_f64;

    let mut p2_x = 0.0_f64;
    let mut p2_y = 0.0_f64;

    // here we scale theta back to "base 180", if theta scale factor was > 1
    let theta = (theta as f64 / theta_axis_scale_factor as f64).round();
    let theta_axis_size = 180;

    let alpha = theta % 90.0;
    let beta = 90.0 - alpha;

    // special cases - line is parallel to x/y axis
    if theta == 0.0 || theta == 180.0 {
        p1_x = rho.abs();
        p1_y = img_height as f64;

        p2_x = rho.abs();
        p2_y = 0.0;
    } else if theta == 90.0 {
        p1_x = 0.0;
        p1_y = rho.abs();

        p2_x = img_width as f64;
        p2_y = rho.abs();
        // otherwise use law of sines to get lines
    } else if theta > 0.0 && theta < 90.0 {
        // start
        p1_x = 0.0;
        p1_y = rho.abs() / deg2rad(theta as u32, theta_axis_size).sin();

        // end
        p2_x = rho.abs() / deg2rad(beta as u32, theta_axis_size).sin();
        p2_y = 0.0;
    } else if theta > 90.0 && theta < 180.0 {
        // start
        if rho < 0.0 {
            p1_x = rho.abs() / deg2rad(alpha as u32, theta_axis_size).sin();
        } else {
            p1_x = rho.abs() * -1.0 / deg2rad(alpha as u32, theta_axis_size).sin();
        }

        p1_y = 0.0;

        // end
        p2_x = img_width as f64;

        if rho < 0.0 {
            p2_y = (img_width as f64 - p1_x.abs()) * deg2rad(alpha as u32, theta_axis_size).sin()
                / deg2rad(beta as u32, theta_axis_size).sin();
        } else {
            p2_y = (img_width as f64 + p1_x.abs()) * deg2rad(alpha as u32, theta_axis_size).sin()
                / deg2rad(beta as u32, theta_axis_size).sin();
        }
    }

    (
        p1_x.round() as i32,
        p1_y.round() as i32,
        p2_x.round() as i32,
        p2_y.round() as i32,
    )
}

#[inline]
fn scale_rho(rho: f64, rho_axis_size: u32, max_line_length: f64) -> u32 {
    let rho_axis_half = (rho_axis_size as f64 / 2.0).round();
    ((rho * rho_axis_half / max_line_length).round() + rho_axis_half) as u32
}

#[inline]
fn is_edge(pixel: &image::Rgba<u8>) -> bool {
    // channels() is deprecated. We can index directly into the Rgba array.
    let greyscale_value = rgb_to_greyscale(pixel[0], pixel[1], pixel[2]);
    greyscale_value < 1
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

fn hough_transform(
    img: &image::DynamicImage,
    theta_axis_scale_factor: u32,
    rho_axis_scale_factor: u32,
) -> na::DMatrix<u32> {
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
            na::DMatrix::from_element(theta_axis_size as usize, rho_axis_size as usize, 0),
            |mut accu, (theta, rho)| {
                let rho_scaled = scale_rho(rho, rho_axis_size, max_line_length);
                accu[(theta as usize, rho_scaled as usize)] += 1;
                accu
            },
        )
}

const D4X: [i32; 4] = [1, 0, -1, 0];
const D4Y: [i32; 4] = [0, 1, 0, -1];

fn should_detect(x: u32, y: u32, src: &DynamicImage) -> bool {
    return color_dist(&src.get_pixel(x, y), &DETECT_COLOR) <= THRESHOLD;
}

fn dfs(
    x: u32,
    y: u32,
    src: &DynamicImage,
    visited: &mut Vec<Vec<bool>>,
    img: &mut DynamicImage,
) -> u32 {
    // u32 (unsigned 32 bit) wraps to really large when < 0, so no need to check for x < 0 or y < 0
    if x >= src.width() || y >= src.height() || visited[x as usize][y as usize] {
        return 0;
    }

    let mut marked: u32 = 0;
    visited[x as usize][y as usize] = true;
    if should_detect(x, y, src) {
        img.put_pixel(x, y, BLACK);
        marked += 1;
        for r in 0..4 {
            let (nx, ny) = ((x as i32 + D4X[r]) as u32, (y as i32 + D4Y[r]) as u32);
            marked += dfs(nx, ny, src, visited, img);
        }
    }
    return marked;
}

fn get_shapes(src: DynamicImage) -> Vec<DynamicImage> {
    let mut visited = vec![vec![false; src.height() as usize]; src.width() as usize];
    let mut result: Vec<DynamicImage> = vec![];
    for x in 0..src.width() {
        for y in 0..src.height() {
            if !visited[x as usize][y as usize] && should_detect(x, y, &src) {
                let mut img = DynamicImage::new_rgba8(src.width(), src.height());
                // I love double nested loops n^4 yippee
                for x in 0..src.width() {
                    for y in 0..src.height() {
                        img.put_pixel(x, y, WHITE);
                    }
                }
                if dfs(x, y, &src, &mut visited, &mut img) as f64
                    > (src.width() * src.height()) as f64 * MIN_PIXEL_AMOUNT
                {
                    result.push(img);
                }
            }
        }
    }
    return result;
}

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

    let src = image::open(&input_img_path).expect("ERROR: input file not found.");
    let mut shapes: Vec<DynamicImage> = get_shapes(src);

    for (i, img) in shapes.iter_mut().enumerate() {
        img.save(PathBuf::from(args[1].to_string()).join(format!(
            "{0}-{1}.png",
            input_img_path.file_name().unwrap().to_string_lossy(),
            i
        )))
        .unwrap();

        let accu = hough_transform(&img, theta_axis_scale_factor, rho_axis_scale_factor);

        dump_houghspace(
            &accu,
            PathBuf::from(args[1].to_string()).join(format!(
                "{0}-{1}-space.png",
                input_img_path.file_name().unwrap().to_string_lossy(),
                i
            )),
        );

        let max_accumulator_value = matrix_max(&accu).unwrap_or(0);

        dump_line_visualization(
            img,
            &accu,
            theta_axis_scale_factor,
            (max_accumulator_value as i32 - houghspace_filter_offset as i32).max(0) as u32,
            PathBuf::from(args[1].to_string()).join(format!(
                "{0}-{1}-lines.png",
                input_img_path.file_name().unwrap().to_string_lossy(),
                i
            )),
        );
    }
}

// -- utility functions --

// Liang-Barsky function by Daniel White @ http://www.skytopia.com/project/articles/compsci/clipping.html
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
