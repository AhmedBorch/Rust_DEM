use std::fs;
use std::path::Path;
use std::usize;
use image::GrayImage;
use image::Luma;
use image::RgbImage;
use image::Rgb;
use proj4rs;
use proj4rs::proj::Proj;
use colorgrad::Gradient;

fn get_data(content: &str, params: &mut GridParams) -> Vec<Vec<f32>> {
    // getting the grid parameters
    let mut lines = content.lines();
    for _ in 0..6 {
        if let Some(line) = lines.next() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("xllcenter") => params.x_begin = parts.next().and_then(|s| s.parse::<f64>().ok()),
                Some("yllcenter") => params.y_begin = parts.next().and_then(|s| s.parse::<f64>().ok()),
                Some("cellsize") => params.cellsize = parts.next().and_then(|s| s.parse::<f64>().ok()),
                Some("nodata_value") => params.nodata = parts.next().and_then(|s| s.parse::<f64>().ok()),
                _ => continue,
            }
        }
    }

    lines.map(|line| {
        line.split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect()
    }).collect()
}

fn find_min(grid: &Vec<Vec<Option<f32>>>) -> Option<f32> {

    grid.iter()
        .flatten()
        .filter_map(|&x| x)
        .reduce(f32::min)
    
}

fn find_max(grid: &Vec<Vec<Option<f32>>>) -> Option<f32> {
    let mut max_val_x = 0.0;
    let mut max_val_y = 0.0;
    let mut max_elevation = 0.0;

    grid.iter()
        .flatten()
        .filter_map(|&x| x)
        .reduce(f32::max)
}

fn find_max_position(grid: &Vec<Vec<Option<f32>>>) -> Option<(usize, usize, f32)> {
    let mut max_pos = None;
    
    for (row_idx, row) in grid.iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            if let Some(val) = val {
                match max_pos {
                    Some((_, _, current_max)) if val > current_max => {
                        max_pos = Some((row_idx, col_idx, val));
                    }
                    None => {
                        max_pos = Some((row_idx, col_idx, val));
                    }
                    _ => {}
                }
            }
        }
    }
    
    max_pos
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


fn grid_to_colored_image(grid: &Vec<Vec<Option<f32>>> )->RgbImage {
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

    // Draw a black "X" every 50 cells
    for y in (0..height).step_by(20) {
        for x in (0..width).step_by(20) {
            let radius = 2; // Size of the "X"
            for offset in 0..=radius {
                // Draw the diagonal from top-left to bottom-right
                if x + offset < width && y + offset < height {
                    image.put_pixel((x + offset) as u32, (y + offset) as u32, Rgb([0, 0, 0])); // Black
                }
                if x >= offset && y >= offset {
                    image.put_pixel((x - offset) as u32, (y - offset) as u32, Rgb([0, 0, 0])); // Black
                }

                // Draw the diagonal from top-right to bottom-left
                if x >= offset && y + offset < height {
                    image.put_pixel((x - offset) as u32, (y + offset) as u32, Rgb([0, 0, 0])); // Black
                }
                if x + offset < width && y >= offset {
                    image.put_pixel((x + offset) as u32, (y - offset) as u32, Rgb([0, 0, 0])); // Black
                }
            }

            // Compute gradient and draw dashed line
            draw_dashed_line(&mut image, &grid, x, y, cell_size);
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



fn get_matrix_dimensions(matrix: &Vec<Vec<f32>>) -> (usize, usize) {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, |row| row.len());
    (rows, cols)
}

// fn transform_coordinates(grid: &Vec<Vec<f32>>)-> Vec<Vec<f32>> {
fn transform_coordinates(rows_len:usize , cols_len:usize, cellsize:f64 ){


    let lambert93 = Proj::from_proj_string(
        "+proj=lcc +lat_1=49 +lat_2=44 +lat_0=46.5 +lon_0=3 +x_0=700000 +y_0=6600000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    ).unwrap();

    let wgs84 = Proj::from_proj_string("+proj=longlat +datum=WGS84 +no_defs").unwrap();
    // Example: Convert a point (X, Y) in Lambert-93 to (Lat, Lon) in WGS84
    let xllcenter = 929000.0;  // Example X coordinate (meters)
    let yllcenter = 6224001.0; // Example Y coordinate (meters)

    // Calculate points (now using f64)
    let mut points = [
        (xllcenter, yllcenter),                          // origin
        (xllcenter + (rows_len as f64) * cellsize, yllcenter),  // top
        (xllcenter, yllcenter + (cols_len as f64) * cellsize),  // right
        (xllcenter + (rows_len as f64) * cellsize,       // top-right
         yllcenter + (cols_len as f64) * cellsize)
    ];

    // Transform all points
    for point in &mut points {
        proj4rs::transform::transform(&lambert93, &wgs84, point).unwrap();
        // Convert radians to degrees
        point.0 = point.0.to_degrees();
        point.1 = point.1.to_degrees();
    }

    // Print results
    println!("Origin: Longitude: {:.6}, Latitude: {:.6}", points[0].0, points[0].1);
    println!("Top: Longitude: {:.6}, Latitude: {:.6}", points[1].0, points[1].1);
    println!("Right: Longitude: {:.6}, Latitude: {:.6}", points[2].0, points[2].1);
    println!("Top right: Longitude: {:.6}, Latitude: {:.6}", points[3].0, points[3].1);

}

#[derive(Debug)]
struct GridParams {
    x_begin: Option<f64>, 
    y_begin: Option<f64>,  
    cellsize: Option<f64>,
    nodata: Option<f64>,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            x_begin: None,
            y_begin: None,
            cellsize: None,
            nodata:None,
        }
    }
}

fn main() {
    let file_path = Path::new("./0925_6225/LITTO3D_FRA_0927_6223_20150529_LAMB93_RGF93_IGN69/MNT1m/LITTO3D_FRA_0927_6223_MNT_20150529_LAMB93_RGF93_IGN69.asc");
    // Initialize to None
    let mut grid: Option<Vec<Vec<f32>>> = None; 
    let mut params = GridParams::default(); 

    match fs::read_to_string(file_path) {
        Ok(content) => {
            grid = Some(get_data(content.as_str(),&mut params));
            println!("Params: {:?}", params);
        },
        
        Err(e) => eprintln!("Failed to read data file: {}", e), // Print error to stderr
    }

    // check if match has a value
    match grid {
        // Some(image) => println!("Transformed content: {:?}", image),
        Some(grid_data) =>
        {
            // replace the -99999 value (or any other nodata value) with None
            let transformed_grid = replace_nodata(&grid_data);

            let (rows, cols) = get_matrix_dimensions(&grid_data);
            println!("Rows: {}, Columns: {}", rows, cols);  // Output: Rows: 2, Columns: 3
            if let Some((row, col, max_val)) = find_max_position(&transformed_grid) {
                println!("Max value {} at row {}, column {}", max_val, row, col);
                    transform_coordinates(rows,cols,1.0);
                }
                // Output: Max value 4.8 at row 2, column 0
            else {
                println!("Grid contains no valid values");
            }
            
            let img_col = grid_to_colored_image(&transformed_grid);
            img_col.save("colored.png").unwrap();

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
        let transformed_grid = replace_nodata(&grid);
        let img = grid_to_colored_image(&transformed_grid);
        let output_path = "test_colored.png";
        img.save(output_path).expect("Failed to save image");
        assert!(std::path::Path::new(output_path).exists(), "Output file should be created");

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
        let img = grid_to_colored_image(&grid);
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
    fn test_find_max_position() {
        let grid = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(4.0), Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0), Some(9.0)],
        ];
        let max_pos = find_max_position(&grid);
        assert_eq!(max_pos, Some((2, 2, 9.0)), "Max value should be at (2, 2) with value 9.0");
    }

    #[test]
    fn test_replace_nodata() {
        let grid = vec![
            vec![-99999.0, 2.0, 3.0],
            vec![4.0, -99999.0, 6.0],
            vec![7.0, 8.0, -99999.0],
        ];
        let replaced_grid = replace_nodata(&grid);
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

    #[test]
    fn test_transform_coordinates() {
        let rows = 100;
        let cols = 100;
        let cellsize = 1.0;
        transform_coordinates(rows, cols, cellsize);
        // No assertion here, just ensuring the function runs without panicking
    }
}