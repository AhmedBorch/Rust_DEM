use std::fs;
use std::path::Path;
use std::usize;
use image::GrayImage;
use image::Luma;
use image::RgbImage;
use image::Rgb;
use std::env;

fn get_data(content: &str, params: &mut GridParams) -> Vec<Vec<f32>> {
    // getting the grid parameters
    let mut lines = content.lines();
    // the 6 first lines have the paramters
    for _ in 0..6 {
        if let Some(line) = lines.next() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("cellsize") => params.cellsize = parts.next().and_then(|s| s.parse::<f32>().ok()),
                Some("nodata_value") => params.nodata = parts.next().and_then(|s| s.parse::<f32>().ok()),
                _ => continue,
            }
        }
    }
    // the rest of the lines has the data, extract it to a matrix form
    lines.map(|line| {
        // data values are split by whitespaces
        line.split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect()
    }).collect()
}

fn find_min(grid: &Vec<Vec<Option<f32>>>) -> Option<f32> {
    // iterate through the matrix and get the minimum value
    grid.iter()
        .flatten()
        .filter_map(|&x| x)
        .reduce(f32::min)
    
}

fn find_max(grid: &Vec<Vec<Option<f32>>>) -> Option<f32> {
    // iterate through the matrix and get the maximum value
    grid.iter()
        .flatten()
        .filter_map(|&x| x)
        .reduce(f32::max)
}


fn replace_nodata(grid: &Vec<Vec<f32>>, nodata_val:Option<f32>) -> Vec<Vec<Option<f32>>> {
    // the nodata value is passed, otherwise the default is -99999.0
    let nodata = nodata_val.unwrap_or(-99999.0);
    // iterate through the matrix and if the value is -99999.0 (using float comparision), put None
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|&x| if (x - nodata).abs() < f32::EPSILON { None } else { Some(x) }) // Replace NaNs
                .collect()
        })
        .collect()
}

fn compute_hillshade(
    grid: &[Vec<Option<f32>>],
    y: usize,
    x: usize,
    cell_size: f32,
    azimuth: f32,
    altitude: f32,
) -> f32 {
    // get the dimensions
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


fn grid_to_colored_image(grid: &Vec<Vec<Option<f32>>>,cellsize:Option<f32> )->(RgbImage, RgbImage) {
    // get image dimensions
    let height = grid.len();
    let width = if height > 0 { grid[0].len() } else { 0 };

    let mut image = RgbImage::new(width as u32, height as u32);

    // Light parameters default values
    let azimuth = 315.0f32.to_radians(); // Light direction (NW)
    let altitude = 45.0f32.to_radians(); // Sun angle
    // get the cellsize value for the hillshade algorithm, 1.0 is the default value
    let cell_size = cellsize.unwrap_or(1.0);
    // Find min and max values ignoring None
    let max_depth = find_min(&grid);
    let min_depth = find_max(&grid);

    // if there is no valid data points, return
    let (max_val, min_val) = match (max_depth, min_depth) {
        (Some(max), Some(min)) => (max, min),
        _ => {
            eprintln!("No valid data points. Returning blank images.");
            return (image.clone(),image);
        }
    };

    // loop through all the points and assign pixel values
    for (y, row) in grid.iter().enumerate() {
        for (x, value_opt) in row.iter().enumerate() {
            let pixel = match value_opt {
                Some(value) => {
                    // Handle case where all values are the same
                    let t = if (max_val - min_val).abs() < f32::EPSILON {
                        0.5
                    } else {
                        // normalize the values, to have something in the [0,1] range
                        (value - min_val) / (max_val - min_val)
                    };
                    
                    // Convert to color gradient (blue -> green -> red)
                    let (r, g, b) = hsl_to_rgb(
                        240.0 * (t),  // Hue from 240° (blue) to 0° (red), map [0,1] range to [0,240]
                        0.8,                // Full saturation
                        0.5                 // Medium lightness
                    );
                    // Calculate hillshade
                    let hillshade = compute_hillshade(&grid, y, x, cell_size, azimuth, altitude);

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

    //clone the image to add the feature
    let mut x_marked_image = image.clone();    // Draw a black "X" every 50 cells
    for y in (0..height).step_by(20) {
        for x in (0..width).step_by(20) {
            let radius = 2; // Size of the "X"
            for offset in 0..=radius {
                // Draw the diagonal from top-left to bottom-right
                if x + offset < width && y + offset < height {
                    x_marked_image.put_pixel((x + offset) as u32, (y + offset) as u32, Rgb([0, 0, 0])); // Black
                }
                if x >= offset && y >= offset {
                    x_marked_image.put_pixel((x - offset) as u32, (y - offset) as u32, Rgb([0, 0, 0])); // Black
                }

                // Draw the diagonal from top-right to bottom-left
                if x >= offset && y + offset < height {
                    x_marked_image.put_pixel((x - offset) as u32, (y + offset) as u32, Rgb([0, 0, 0])); // Black
                }
                if x + offset < width && y >= offset {
                    x_marked_image.put_pixel((x + offset) as u32, (y - offset) as u32, Rgb([0, 0, 0])); // Black
                }
            }

            // Compute gradient and draw dashed line
            draw_dashed_line(&mut x_marked_image, &grid, x, y, cell_size);
        }
    }

    return (image,x_marked_image);
}

fn grid_to_image(grid: &Vec<Vec<Option<f32>>> )->GrayImage {
    // get image dimensions
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
                        // normalize the values
                        (value - min_val) / range
                    };
                    let inverted = 1.0 - normalized;
                    // clip to [0, 255]
                    (inverted * 255.0).clamp(0.0, 255.0) as u8
                }
                None => 0u8, // No-data values are black
            };

            image.put_pixel(x as u32, y as u32, Luma([pixel_value]));
        }
    }
    return image;
}

fn draw_dashed_line(
    image: &mut RgbImage,
    grid: &[Vec<Option<f32>>],
    start_x: usize,
    start_y: usize,
    cell_size: f32,
) {
    let mut x = start_x;
    let mut y = start_y;
    let mut dash_on = true; // Toggle for dashed line
    let mut iterations = 0; // Counter for iterations

    loop {
        // Check boundaries
        if x == 0 || x >= grid[0].len() - 1 || y == 0 || y >= grid.len() - 1 {
            break;
        }

        // Stop after 100 iterations
        if iterations >= 10000 {
            break;
        }

        // Get current value
        let current_value = match grid[y][x] {
            Some(value) => value,
            None => break,
        };

        // Check if current cell is a local minimum
        let neighbors = [
            grid[y - 1][x],     // Top
            grid[y + 1][x],     // Bottom
            grid[y][x - 1],     // Left
            grid[y][x + 1],     // Right
            grid[y - 1][x - 1], // Top-left
            grid[y - 1][x + 1], // Top-right
            grid[y + 1][x - 1], // Bottom-left
            grid[y + 1][x + 1], // Bottom-right
        ];

        if neighbors.iter().all(|&n| n.unwrap_or(f32::INFINITY) > current_value) {
            break; // Stop at local minimum
        }

        // Compute gradient direction
        let dzdx = (grid[y][x + 1].unwrap_or(current_value) - grid[y][x - 1].unwrap_or(current_value))
            / (2.0 * cell_size);
        let dzdy = (grid[y + 1][x].unwrap_or(current_value) - grid[y - 1][x].unwrap_or(current_value))
            / (2.0 * cell_size);

        let magnitude = (dzdx.powi(2) + dzdy.powi(2)).sqrt();
        let direction_x = (-dzdx / magnitude).round() as isize;
        let direction_y = (-dzdy / magnitude).round() as isize;

        // Draw pixel if dash is "on"
        if dash_on {
            image.put_pixel(x as u32, y as u32, Rgb([255, 0, 0])); // Red dashed line
        }

        // Toggle dash
        dash_on = !dash_on;

        // Move to next cell in gradient direction
        x = (x as isize + direction_x) as usize;
        y = (y as isize + direction_y) as usize;

        // Increment iteration counter
        iterations += 1;
    }
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


#[derive(Debug)]
struct GridParams {
    cellsize: Option<f32>,
    nodata: Option<f32>,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            cellsize: None,
            nodata:None,
        }
    }
}

fn main() {
    // Collect command-line arguments
    let args: Vec<String> = env::args().collect();

    // Check if a file path was provided
    if args.len() < 2 {
        eprintln!("Usage: {} <file_path>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    println!("Processing file: {}", file_path);
    
    let file_path = Path::new(file_path);
    // Initialize to None
    let mut grid: Option<Vec<Vec<f32>>> = None; 
    let mut params = GridParams::default(); 

    match fs::read_to_string(file_path) {
        Ok(content) => {
            grid = Some(get_data(content.as_str(),&mut params));
            println!("These are the parameters from the data: {:?}", params);
        }, 
        Err(e) => eprintln!("Failed to read data file: {}", e), // Print error to stderr
    }

    // check if grid has values
    match grid {
        // Some(image) => println!("Transformed content: {:?}", image),
        Some(grid_data) =>
        {
            // replace the -99999 value (or any other nodata value) with None
            let transformed_grid = replace_nodata(&grid_data,params.nodata);
            
            //save the greyscale image
            let img = grid_to_image(&transformed_grid);
            img.save("grayscale.png").unwrap();
            let (img_col,img_feature) = grid_to_colored_image(&transformed_grid,params.cellsize);
            img_col.save("colored.png").unwrap();
            img_feature.save("gradient.png").unwrap();

        }
        None => println!("No image data available."),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_grayscale_image() {
        let filepath = "./0925_6225/LITTO3D_FRA_0925_6225_20150529_LAMB93_RGF93_IGN69/MNT1m/LITTO3D_FRA_0925_6225_MNT_20150529_LAMB93_RGF93_IGN69.asc";
        let mut params = GridParams::default();
        let content = fs::read_to_string(filepath).expect("Failed to read file");
        let grid = get_data(&content, &mut params);
        let transformed_grid = replace_nodata(&grid, params.nodata);
        let img = grid_to_image(&transformed_grid);
        let output_path = "test_grayscale.png";
        img.save(output_path).expect("Failed to save grayscale image");
        assert!(std::path::Path::new(output_path).exists(), "Grayscale image file should be created");

        // Clean up the output file
        fs::remove_file(output_path).expect("Failed to remove test output file");
    }

    #[test]
    fn test_generate_colored_image() {
        let grid = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(4.0), Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0), Some(9.0)],
        ];
        let (img, _) = grid_to_colored_image(&grid, Some(1.0));
        let output_path = "test_colored_image.png";
        img.save(output_path).expect("Failed to save colored image");
        assert!(std::path::Path::new(output_path).exists(), "Colored image file should be created");

        // Clean up the output file
        fs::remove_file(output_path).expect("Failed to remove test output file");
    }

    #[test]
    fn test_find_min() {
        let grid = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(4.0), Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0), Some(9.0)],
        ];
        let min_val = find_min(&grid);
        assert_eq!(min_val, Some(1.0), "Min value should be 1.0");
    }

    #[test]
    fn test_find_max() {
        let grid = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(4.0), Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0), Some(9.0)],
        ];
        let max_val = find_max(&grid);
        assert_eq!(max_val, Some(9.0), "Max value should be 9.0");
    }

    #[test]
    fn test_replace_nodata() {
        let grid = vec![
            vec![-99999.0, 2.0, 3.0],
            vec![4.0, -99999.0, 6.0],
            vec![7.0, 8.0, -99999.0],
        ];
        let replaced_grid = replace_nodata(&grid, None);
        assert_eq!(
            replaced_grid,
            vec![
                vec![None, Some(2.0), Some(3.0)],
                vec![Some(4.0), None, Some(6.0)],
                vec![Some(7.0), Some(8.0), None],
            ],
            "No-data values should be replaced with None"
        );
    }

    #[test]
    fn test_compute_hillshade() {
        let grid = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(4.0), Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0), Some(9.0)],
        ];
        let hillshade = compute_hillshade(&grid, 1, 1, 1.0, 315.0f32.to_radians(), 45.0f32.to_radians());
        assert!(hillshade >= 0.0 && hillshade <= 1.0, "Hillshade should be between 0.0 and 1.0");
    }

    #[test]
    fn test_hsl_to_rgb() {
        let (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.5);
        assert_eq!((r, g, b), (255, 0, 0), "HSL(0, 1, 0.5) should convert to RGB(255, 0, 0)");
    }
}