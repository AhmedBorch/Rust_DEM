use std::fs;
use std::path::Path;
use image::GrayImage;
use image::Luma;
use image::RgbImage;
use image::Rgb;

fn get_data(content: &str) -> Vec<Vec<f32>> {
    let mut grid: Vec<Vec<f32>> = Vec::new(); // This will store the grid
    // let mut a = 0;
    for line in content.lines().skip(6) { // Skip first 4 lines, start from 5th
        // println!("{}", line);
        // a = a+1;
        // if a==3{
        //     break;
        // }
        let row: Vec<f32> = line.split_whitespace() // Split the line by spaces
            .filter_map(|s| s.parse::<f32>().ok()) // Parse each value into f32, ignore errors
            .collect(); // Collect into a Vec<f32>
        grid.push(row); // Add the row to the grid
    }
    // let grey_image = vec![1,2];
    return grid
}

fn grid_to_image(grid: Vec<Vec<Option<f32>>> )->GrayImage {
    let height = grid.len();
    let width = if height > 0 { grid[0].len() } else { 0 };

    let mut image = GrayImage::new(width as u32, height as u32);

    let max_depth = find_min(&grid);
    let min_depth = find_max(&grid);

    let (max_val, min_val) = match (max_depth, min_depth) {
        (Some(max), Some(min)) => (max, min),
        _ => {
            eprintln!("No valid data points. Returning blank image.");
            return image;
        }
    };

    for (y, row) in grid.iter().enumerate() {
        for (x, value_opt) in row.iter().enumerate() {
            let pixel_value = match value_opt {
                Some(value) => {
                    // Handle case where all values are the same
                    let range = max_val - min_val;
                    let normalized = if range.abs() < f32::EPSILON {
                        0.0
                    } else {
                        (value - min_val) / range
                    };
                    let inverted = 1.0 - normalized;
                    (inverted * 255.0).clamp(0.0, 255.0) as u8
                }
                None => 0u8, // No-data values are black
            };

            image.put_pixel(x as u32, y as u32, Luma([pixel_value]));
        }
    }
    // println!("this is the deepeset point: {:?}",max_depth);
    // println!("this is the sallowest point: {:?}",min_depth);
    // let width = image.width() as usize;
    // println!("Raw pixel data: {:?}", &image.as_raw()[..width]);
    return image;
}

fn compute_hillshade(
    grid: &[Vec<Option<f32>>],
    y: usize,
    x: usize,
    cell_size: f32,
    azimuth: f32,
    altitude: f32,
) -> f32 {
    let height = grid.len();
    let width = if height > 0 { grid[0].len() } else { 0 };

    // Check boundaries and valid neighbors
    if y == 0 || y >= height - 1 || x == 0 || x >= width - 1 {
        return 1.0;
    }

    // Collect 3x3 neighborhood values
    let mut neighborhood = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let yi = y + i - 1;
            let xi = x + j - 1;
            match grid.get(yi).and_then(|row| row.get(xi)).copied().flatten() {
                Some(v) => neighborhood[i][j] = v,
                None => return 1.0,
            }
        }
    }

    // Horn's formula for slope calculation
    let dzdx = neighborhood[0][2] + 2.0 * neighborhood[1][2] + neighborhood[2][2]
        - neighborhood[0][0] - 2.0 * neighborhood[1][0] - neighborhood[2][0];
    let dzdy = neighborhood[2][0] + 2.0 * neighborhood[2][1] + neighborhood[2][2]
        - neighborhood[0][0] - 2.0 * neighborhood[0][1] - neighborhood[0][2];
    
    let dzdx = dzdx / (8.0 * cell_size);
    let dzdy = dzdy / (8.0 * cell_size);

    let slope = (dzdx.powi(2) + dzdy.powi(2)).sqrt().atan();
    let aspect = (-dzdy).atan2(-dzdx); // Correct aspect calculation

    // Hillshade calculation
    let hillshade = altitude.cos() * slope.cos()
        + altitude.sin() * slope.sin() * (azimuth - aspect).cos();

    hillshade.max(0.0).min(1.0)
}


fn grid_to_colored_image(grid: Vec<Vec<Option<f32>>> )->RgbImage {
    let height = grid.len();
    let width = if height > 0 { grid[0].len() } else { 0 };

    let mut image = RgbImage::new(width as u32, height as u32);

    // Light parameters (adjust these to change lighting)
    let azimuth = 315.0f32.to_radians(); // Light direction (NW)
    let altitude = 45.0f32.to_radians(); // Sun angle
    let cell_size=1.0;
    // Find min and max values ignoring None
    let max_depth = find_min(&grid);
    let min_depth = find_max(&grid);

    let (max_val, min_val) = match (max_depth, min_depth) {
        (Some(max), Some(min)) => (max, min),
        _ => {
            eprintln!("No valid data points. Returning blank image.");
            return image;
        }
    };

    for (y, row) in grid.iter().enumerate() {
        for (x, value_opt) in row.iter().enumerate() {
            let pixel = match value_opt {
                Some(value) => {
                    // Handle case where all values are the same
                    let t = if (max_val - min_val).abs() < f32::EPSILON {
                        0.5
                    } else {
                        (value - min_val) / (max_val - min_val)
                    };
                    
                    // Convert to color gradient (blue -> green -> red)
                    let (r, g, b) = hsl_to_rgb(
                        240.0 * (t),  // Hue from 240° (blue) to 0° (red)
                        0.8,                // Full saturation
                        0.5                 // Medium lightness
                    );
                    // Calculate hillshade
                    let hillshade = compute_hillshade(&grid, y, x, cell_size, azimuth, altitude);
                    // Rgb([r, g, b])

                    // Combine color with hillshade
                    Rgb([
                        (f32::from(r) * hillshade) as u8,
                        (f32::from(g) * hillshade) as u8,
                        (f32::from(b) * hillshade) as u8,
                    ])
                }
                None => Rgb([54, 69, 79]), // No-data values are grey
            };
            image.put_pixel(x as u32, y as u32, pixel);
        }
    }
    return image;
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match h {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

fn find_min(grid: &Vec<Vec<Option<f32>>>) -> Option<f32> {
    grid.iter()
        .flatten()
        .filter_map(|&x| x)
        .reduce(f32::min)
}

fn find_max(grid: &Vec<Vec<Option<f32>>>) -> Option<f32> {
    grid.iter()
        .flatten()
        .filter_map(|&x| x)
        .reduce(f32::max)
}


fn replace_nodata(grid: &Vec<Vec<f32>>) -> Vec<Vec<Option<f32>>> {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|&x| if (x + 99999.0).abs() < 0.0001 { None } else { Some(x) }) // Replace NaNs
                .collect()
        })
        .collect()
}

fn main() {
    let file_path = Path::new("/home/ahmed/Desktop/2nd_semester/RUST_COURSE/project/data/0925_6225/LITTO3D_FRA_0925_6225_20150529_LAMB93_RGF93_IGN69/MNT1m/LITTO3D_FRA_0925_6225_MNT_20150529_LAMB93_RGF93_IGN69.asc");

    // Initialize to None
    let mut grid: Option<Vec<Vec<f32>>> = None; 

    match fs::read_to_string(file_path) {
        Ok(content) => {
            grid = Some(get_data(content.as_str()));
        },
        
        Err(e) => eprintln!("Failed to read data file: {}", e), // Print error to stderr
    }

    // check if match has a value
    match grid {
        // Some(image) => println!("Transformed content: {:?}", image),
        Some(grid_data) =>
        {
            let transformed_grid = replace_nodata(&grid_data);
            //  access the first row
            // println!("Grid Sample: {:?}", transformed_grid[0]); // First row

            // let img = grid_to_image(transformed_grid);
            // img.save("greyscale.png").unwrap();
            let img_col = grid_to_colored_image(transformed_grid);
            img_col.save("colored.png").unwrap();
        }
        None => println!("No image data available."),
    }
}


// 1. fetch values
// 2. transform to Grey scale points
// 3. construct and show the image


