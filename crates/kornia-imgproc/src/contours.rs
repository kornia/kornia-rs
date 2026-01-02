use kornia_image::{Image, ImageError};
use std::collections::HashMap;

/// Mode of the contour retrieval algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Retrieves only the extreme outer contours.
    External,
    /// Retrieves all of the contours without establishing any hierarchical relationships.
    List,
    /// Retrieves all of the contours and organizes them into a two-level hierarchy.
    Ccomp,
    /// Retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    Tree,
}

/// The approximation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourApproximationMode {
    /// Stores absolutely all the contour points.
    None,
    /// Compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    Simple,
}

/// A point in 2D space.
pub type Point = [i32; 2];

/// A 2D contour is a sequence of points.
pub type Contour = Vec<Point>;

/// Hierarchy information for a contour.
/// [Next, Previous, First_Child, Parent]
pub type Vec4i = [i32; 4];

/// Finds contours in a binary image.
///
/// The function retrieves contours from the binary image using the algorithm
/// [Suzuki85](https://www.sciencedirect.com/science/article/pii/0734189X85900167).
/// The image is treated as a binary image. Non-zero pixels are treated as 1s.
/// Zero pixels remain 0s, so the image is treated as binary .
///
/// # Arguments
///
/// * `src` - The input image (8-bit single-channel).
/// * `mode` - Contour retrieval mode.
/// * `method` - Contour approximation method.
///
/// # Returns
///
/// A tuple containing:
/// * A list of contours.
/// * A list of hierarchy information (optional, depending on mode).
pub fn find_contours<A>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
) -> Result<(Vec<Contour>, Vec<Vec4i>), ImageError>
where
    A: kornia_image::allocator::ImageAllocator,
{
    let width = src.width();
    let height = src.height();

    // 1. Pad the image with 1 pixel border to handle boundary conditions easier
    // We used a simplified approach treating the image as if it was padded
    // or by checking bounds carefully. For this implementation, let's use a copy with padding
    // to match the original algorithm description which assumes 0-border.
    // However, for performance we might want to avoid full copy.
    // Let's stick to the algorithm description: "The picture frame is denoted by F".
    // We can assume the input image is surrounded by 0s.

    // Using a padded image for the algorithm to operate on.
    // +2 for padding on both sides.
    let mut image_padded = vec![0i16; (width + 2) * (height + 2)];

    // Copy src to image_padded (converting to i16 for markers).
    // Original algorithm uses:
    // 1 for object (non-zero in src)
    // 0 for background
    // And modifies values during execution.
    // Negative values are used for visited markers.
    // LNBD (Last Non-Zero Border found) logic.

    for y in 0..height {
        for x in 0..width {
            let val = src.get([x, y]).unwrap_or(&0);
            if *val > 0 {
                image_padded[(y + 1) * (width + 2) + (x + 1)] = 1;
            }
        }
    }

    let stride = width + 2;
    let mut contours = Vec::new();
    let mut hierarchy = Vec::new();

    // Directions: 0:E, 1:SE, 2:S, 3:SW, 4:W, 5:NW, 6:N, 7:NE
    // (dx, dy)
    let directions = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    
let mut lnbd: i16 = 1;
let mut nbd: i16 = 1;
let mut nbd_to_index: std::collections::HashMap<i16, usize> = std::collections::HashMap::new();
let mut parent_map: Vec<Option<usize>> = Vec::new();
    for y in 1..=height {
        lnbd = 1;
        for x in 1..=width {
            let idx = y * stride + x;
            let f_xy = image_padded[idx]; // Current pixel value

            let is_start_outer = f_xy == 1 && image_padded[idx - 1] == 0;
            let is_start_hole = f_xy >= 1 && image_padded[idx + 1] == 0;

            if is_start_outer || is_start_hole {
                // Precedence: Outer > Hole
                let is_outer = is_start_outer;
                 && is_start_hole; // Actually if is_outer is true, we process as outer.
                
                // 2.1 Decide the starting direction
                let mut from = if is_outer { 7 } else { 3 };

                // 2.2 Find the starting point of the border
                let mut x_curr = x;
                let mut y_curr = y;
                let mut dir_curr = from;
                let mut found_start = false;
                
                // Helper to get neighbor value
                let get_val = |r: usize, c: usize, img: &Vec<i16>| -> i16 {
                     img[r * stride + c]
                };

                // Search for non-zero neighbor
                let start_idx = if is_outer { 7 } else { 3 }; // Relative directions
                
                let mut first_neighbor_idx = None;
                
                for k in 0..8 {
                    let d = (start_idx + k) % 8;
                    let (dx, dy) = directions[d];
                    let nx_i = x as isize + dx; let ny_i = y as isize + dy; if nx_i < 0 || ny_i < 0 { continue; } let nx = nx_i as usize; let ny = ny_i as usize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    
                    if get_val(ny, nx, &image_padded) != 0 {
                        first_neighbor_idx = Some(d);
                        break;
                    }
                }

                if first_neighbor_idx.is_none() {
                    // Isolated point.
                    image_padded[idx] = -nbd;
                    // Add point to contour ?
                    // Should we add it? Yes.
                    if method == ContourApproximationMode::None {
                         contours.push(vec![[x as i32 - 1, y as i32 - 1]]);
                    } 
                    // Hierarchy update could happen here too.
                    
                    if f_xy != 0 && f_xy != 1 {
                        lnbd = f_xy.abs();
                    }
                    continue;
                }
                
                // We found a border.
                nbd += 1;
                let contour_idx = contours.len();
                nbd_to_index.insert(nbd, contour_idx);
                
                let mut current_contour = Vec::new(); // Store points
                
                let dir_start = first_neighbor_idx.unwrap();
                let (dx, dy) = directions[dir_start];
                let mut x2 = (x as isize + dx) as usize;
                let mut y2 = (y as isize + dy) as usize;
                
                // 3.2
                // (x2, y2) is the first non-zero neighbor.
                
                dir_curr = (dir_start + 4) % 8; // Back towards (x,y) from (x2,y2) 
                x_curr = x2;
                y_curr = y2;
                
                // Add start point
                 current_contour.push([x as i32 - 1, y as i32 - 1]);
                
                // Hierarchy building
                // Determine parent
                // Parent is always the contour associated with LNBD (unless LNBD=1).
                let parent_idx = if lnbd > 1 {
                    nbd_to_index.get(&lnbd).cloned()
                } else {
                    None // Root
                };
                
                parent_map.push(parent_idx);
                
                // Update siblings/children of parent... (Do this after or during?)
                
                // 3.3 Loop
                loop {
                    // Search for next non-zero pixel
                    let mut found_next = false;
                    let mut next_dir = 0;
                    let mut x3 = 0;
                    let mut y3 = 0;
                    
                     for k in 0..8 {
                        let d = (dir_curr + k) % 8;
                        let (dx, dy) = directions[d];
                        let nx = (x_curr as isize + dx) as usize;
                        let ny = (y_curr as isize + dy) as usize;
                        
                        if get_val(ny, nx, &image_padded) != 0 {
                            x3 = nx;
                            y3 = ny;
                            next_dir = d;
                            found_next = true;
                            break;
                        }
                    }
                    
                    if !found_next {
                         // Should not happen for connected component unless isolated, but isolated is handled before.
                         break;
                    }

                     let idx_curr = y_curr * stride + x_curr;
                     
                     if method == ContourApproximationMode::None {
                        current_contour.push([x_curr as i32 - 1, y_curr as i32 - 1]);
                     } else {
                         // Simple approximation
                         // Keep point if direction changes.
                         if current_contour.len() > 1 {
                             let last = current_contour.last().unwrap();
                             let prev = current_contour[current_contour.len() - 2];
                             let dx1 = last[0] - prev[0];
                             let dy1 = last[1] - prev[1];
                             let dx2 = (x_curr as i32 - 1) - last[0];
                             let dy2 = (y_curr as i32 - 1) - last[1];
                             
                             if dx1 * dy2 == dx2 * dy1 {
                                 // Collinear, replace last
                                 current_contour.pop();
                             }
                         }
                         current_contour.push([x_curr as i32 - 1, y_curr as i32 - 1]);
                     }
                     
                     // Mark current pixel
                     if image_padded[idx_curr] != -nbd {
                         image_padded[idx_curr] = nbd; // Mark as part of this contour
                         // Suzuki: "If f(x, y+1) = 0 then f(x,y) = -NBD"
                         // This check is performed in the PADDED coordinate system.
                         // idx_curr is (y_curr * stride + x_curr)
                         // Plus stride is y_curr + 1
                         if image_padded[idx_curr + stride] == 0 {
                             image_padded[idx_curr] = -nbd;
                         } else if image_padded[idx_curr] == 1 {
                             image_padded[idx_curr] = nbd;
                         }
                     }
                     
                     // Termination check
                     if x_curr == x && y_curr == y && x3 == x2 && y3 == y2 {
                         break;
                     }
                     
                     dir_curr = (next_dir + 4 + 1) % 8;
                     x_curr = x3;
                     y_curr = y3;
                }
                
                contours.push(current_contour);
            }

            // Update LNBD
             let idx_curr = y * stride + x;
             let val = image_padded[idx_curr];
            if  val != 0 && val != 1 {
                 lnbd = val.abs();
            }
        }
    }
    
    // Construct hierarchy
    // Vec4i: [Next, Previous, First_Child, Parent]
    hierarchy.resize(contours.len(), [-1, -1, -1, -1]);
    
    // Auxiliary map to help building next/prev siblings
    // parent_idx -> last_child_idx
    let mut last_child = std::collections::HashMap::<usize, usize>::new();
    
    for i in 0..contours.len() {
        if let Some(parent) = parent_map[i] {
            // Set Parent of i
            hierarchy[i][3] = parent as i32;
            
             if let Some(&prev) = last_child.get(&parent) {
                 hierarchy[prev][0] = i as i32; // Next of prev is i
                 hierarchy[i][1] = prev as i32; // Prev of i is prev
             } else {
                 hierarchy[parent][2] = i as i32; // First child of parent is i
             }
             last_child.insert(parent, i);
        }
    }
    
    // Filter based on mode
    match mode {
        RetrievalMode::External => {
             let mut new_contours = Vec::new();
             let mut new_hierarchy = Vec::new();
             // Keep only contours with no parent (parent == -1)
             for i in 0..contours.len() {
                  if hierarchy[i][3] == -1 {
                      new_contours.push(contours[i].clone());
                      new_hierarchy.push([-1, -1, -1, -1]);
                  }
             }
             // Fix Next/Prev for external contours (chain them)
             for i in 0..new_contours.len() {
                 if i + 1 < new_contours.len() {
                     new_hierarchy[i][0] = (i + 1) as i32;
                 }
                 if i > 0 {
                     new_hierarchy[i][1] = (i - 1) as i32;
                 }
             }
             contours = new_contours;
             hierarchy = new_hierarchy;
        },
        RetrievalMode::List => {
             // Establish no hierarchy
              for h in hierarchy.iter_mut() {
                  *h = [-1, -1, -1, -1];
              }
              // Link all as sequence
              for i in 0..contours.len() {
                  if i + 1 < contours.len() {
                      hierarchy[i][0] = (i + 1) as i32;
                  }
                  if i > 0 {
                      hierarchy[i][1] = (i - 1) as i32;
                  }
              }
        },
        RetrievalMode::Ccomp => {
            // 2-level hierarchy:
            // - Top level: Outer boundaries of components.
            // - Second level: Hole boundaries of components.
            // If a hole has another component inside, that component is also top level ??
            // OpenCV description: "The hierarchy is reorganized into two levels.
            // The top level is the external boundaries of the components.
            // The second level is the boundaries of the holes.
            // If there is another contour inside a hole of a connected component, it is still put at the top level."
            
            // We need to iterate and re-link.
            
            // 1. Identify "Top Level" contours.
            // In the full tree, any contour at even depth (0, 2, 4...) is a component boundary (External).
            // Any contour at odd depth (1, 3, 5...) is a hole.
            // But wait, "If there is another contour inside a hole... it is still put at the top level."
            // So:
            // Depth 0 (External) -> Top Level
            // Depth 1 (Hole) -> Second Level (Child of Depth 0)
            // Depth 2 (External inside Hole) -> Top Level ??
            
            // Let's perform a traversal to check depth.
            let mut depths = vec![0; contours.len()];
            
            // Helper to compute depths (can be done by traversing hierarchy)
            // But efficient way: traverse logic.
            // Or just follow parent pointers.
            for i in 0..contours.len() {
                let mut d = 0;
                let mut p = hierarchy[i][3];
                while p != -1 {
                    d += 1;
                    p = hierarchy[p as usize][3];
                }
                depths[i] = d;
            }
            
            // Rebuild hierarchy
            let mut new_hierarchy = vec![[-1i32; 4]; contours.len()];
             // We need to link all Top Levels (Depth even) together as siblings.
             // And link holes (Depth odd) to their parents (which must be Top Level).
             // And link holes of same parent together.
             
             // BUT, if Depth 2 is Top Level, who is its parent? 
             // "It is still put at the top level." -> Parent is -1.
             
             let mut last_top: i32 = -1;
             
             // We need to process in order of appearance in `contours`?
             // Or we can just iterate.
             
             for i in 0..contours.len() {
                 if depths[i] % 2 == 0 {
                     // Top Level
                     new_hierarchy[i][3] = -1; // No parent
                     
                     // Link to previous top level
                     if last_top != -1 {
                         new_hierarchy[last_top as usize][0] = i as i32;
                         new_hierarchy[i][1] = last_top;
                     }
                     last_top = i as i32;
                     
                     // Handle children (holes)
                     // Original children of this contour i in the tree might be holes (Depth d+1)
                     // We need to link them as children of i.
                     // But we only want direct children? 
                     // In `hierarchy`, `i` has `First_Child`.
                     
                     let mut child = hierarchy[i][2];
                     let mut last_hole: i32 = -1;
                     
                     while child != -1 {
                         // child is Depth d+1 (Odd) -> Hole
                         new_hierarchy[i][2] = child; // Set as first child (will be overwritten if we iterate siblings, so we need to set only once? No, we need First Child of i to point to first hole)
                         
                         // Wait, we need to link all holes of `i`.
                         // The holes are siblings in the original hierarchy?
                         // Yes.
                         // So we just preserve the sibling chain of the holes?
                         // But what if a hole has a child (Depth d+2)? 
                         // That child becomes Top Level. So it is not a child of the hole in CCOMP.
                         // So the hole should have NO child in CCOMP.
                         new_hierarchy[child as usize][3] = i as i32; // Parent is i
                         new_hierarchy[child as usize][2] = -1; // No child
                         
                         // Link siblings?
                         // The original validation is: Are all siblings of `child` also holes?
                         // Yes, siblings share the same parent `i`.
                         
                         // We need to maintain the sibling chain for the holes.
                         // `child` in original hierarchy handles this. 
                         // We just need to ensure we don't break it?
                         // Actually, we are rewriting `new_hierarchy`.
                         
                         // Let's traverse the sibling list of `child` in original hierarchy.
                         // All these are holes.
                         
                         let mut curr_hole = child;
                         let mut prev_hole = -1;
                         
                         while curr_hole != -1 {
                                         if curr_hole < 0 || curr_hole as usize >= new_hierarchy.len() { break; }
                             new_hierarchy[curr_hole as usize][3] = i as i32;
                             new_hierarchy[curr_hole as usize][2] = -1; // No child
                             
                             if prev_hole != -1 {
                                 new_hierarchy[prev_hole as usize][0] = curr_hole;
                                 new_hierarchy[curr_hole as usize][1] = prev_hole;
                             } else {
                                 // First hole
                                 new_hierarchy[i][2] = curr_hole;
                             }
                             
                             prev_hole = curr_hole;
            if curr_hole < 0 || curr_hole as usize >= hierarchy.len() { break; } curr_hole = hierarchy[curr_hole as usize][0]; // Next sibling
                         
                         // We processed all children of i.
                         break; 
                     }
                 }
             }
             
             hierarchy = new_hierarchy;
        },
        RetrievalMode::Tree => {
            // Full hierarchy, already built.
        }
    }
    
    Ok((contours, hierarchy))
}
