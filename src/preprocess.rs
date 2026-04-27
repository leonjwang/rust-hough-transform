use image::{DynamicImage, GenericImage, Rgba};
use imageproc::drawing::Canvas;

const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);
const BLACK: Rgba<u8> = Rgba([0, 0, 0, 255]);

const DETECT_COLOR: Rgba<u8> = Rgba([180, 20, 40, 255]); // TODO: Dynamicly detect this from image
const THRESHOLD: f64 = 50.0;

const MIN_PIXEL_AMOUNT: f64 = 0.01;

const D4X: [i32; 4] = [1, 0, -1, 0];
const D4Y: [i32; 4] = [0, 1, 0, -1];

#[inline]
fn color_dist(c1: &Rgba<u8>, c2: &Rgba<u8>) -> f64 {
    return ((c1[0] as f64 - c2[0] as f64).powi(2)
        + (c1[1] as f64 - c2[1] as f64).powi(2)
        + (c1[2] as f64 - c2[2] as f64).powi(2))
    .sqrt();
}

#[inline]
fn should_detect(x: u32, y: u32, src: &DynamicImage) -> bool {
    return color_dist(&src.get_pixel(x, y), &DETECT_COLOR) <= THRESHOLD;
}

// Floodfill
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

pub fn get_shapes(src: DynamicImage) -> Vec<DynamicImage> {
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
