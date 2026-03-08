use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Retrieval mode for contour extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Retrieve only the outermost contours.
    External,
    /// Retrieve all contours without parent/child relationships.
    List,
    /// Retrieve contours in a two-level hierarchy (outer boundaries and holes).
    CComp,
    /// Retrieve all contours and reconstruct full hierarchy.
    Tree,
}

/// Contour approximation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContourApproximationMode {
    /// Store all contour points.
    None,
    /// Compress horizontal, vertical, and diagonal segments.
    Simple,
}

/// Contour represented as image points `(x, y)`.
pub type Contour = Vec<(i32, i32)>;

/// Result of contour extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindContoursResult {
    /// Extracted contours.
    pub contours: Vec<Contour>,
    /// Per-contour hierarchy in OpenCV-compatible layout: `[next, prev, child, parent]`.
    pub hierarchy: Vec<[i32; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContourKind {
    Foreground,
    Hole,
}

#[derive(Debug, Clone)]
struct ContourEntry {
    points: Contour,
    kind: ContourKind,
    component_id: i32,
    parent_hint: Option<usize>,
}

const DIRS_8: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Find contours in a binary image.
///
/// Non-zero pixels are treated as foreground and zero pixels as background.
/// The function follows the high-level semantics of OpenCV `findContours` for
/// retrieval and approximation modes.
///
/// # Arguments
///
/// * `src` - Binary single-channel image (`0` background, non-zero foreground).
/// * `mode` - Contour retrieval mode.
/// * `method` - Contour approximation method.
///
/// # Returns
///
/// A [`FindContoursResult`] containing contour points and hierarchy.
///
/// # Errors
///
/// This function currently does not produce runtime errors and returns `Ok(...)`.
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_imgproc::contours::{find_contours, ContourApproximationMode, RetrievalMode};
///
/// let data = vec![
///     0u8, 0, 0, 0, 0,
///     0, 255, 255, 255, 0,
///     0, 255, 255, 255, 0,
///     0, 255, 255, 255, 0,
///     0, 0, 0, 0, 0,
/// ];
/// let img = Image::<u8, 1, _>::new(ImageSize { width: 5, height: 5 }, data, CpuAllocator)?;
/// let out = find_contours(&img, RetrievalMode::External, ContourApproximationMode::Simple)?;
/// assert_eq!(out.contours.len(), 1);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn find_contours<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
) -> Result<FindContoursResult, ImageError> {
    let width = src.width();
    let height = src.height();
    if width == 0 || height == 0 {
        return Ok(FindContoursResult {
            contours: Vec::new(),
            hierarchy: Vec::new(),
        });
    }

    let fg_mask: Vec<bool> = src.as_slice().iter().map(|&v| v != 0).collect();
    let fg_labels = label_components(&fg_mask, width, height);
    let fg_count = fg_labels
        .iter()
        .copied()
        .max()
        .map_or(0usize, |m| (m as usize) + 1);

    let mut entries = Vec::<ContourEntry>::new();
    let mut fg_comp_to_entry = vec![None; fg_count];

    for comp_id in 0..fg_count {
        let start = find_boundary_start(&fg_labels, width, height, comp_id as i32);
        if let Some(start_idx) = start {
            let contour =
                trace_component_boundary(&fg_labels, width, height, comp_id as i32, start_idx);
            let contour = approximate(contour, method);
            let entry_idx = entries.len();
            entries.push(ContourEntry {
                points: contour,
                kind: ContourKind::Foreground,
                component_id: comp_id as i32,
                parent_hint: None,
            });
            fg_comp_to_entry[comp_id] = Some(entry_idx);
        }
    }

    let mut bg_labels = Vec::<i32>::new();
    let mut bg_touches_border = Vec::<bool>::new();
    let mut bg_comp_to_entry: Vec<Option<usize>> = Vec::new();

    if matches!(
        mode,
        RetrievalMode::List | RetrievalMode::CComp | RetrievalMode::Tree
    ) {
        let bg_mask: Vec<bool> = fg_mask.iter().map(|&is_fg| !is_fg).collect();
        bg_labels = label_components(&bg_mask, width, height);
        let bg_count = bg_labels
            .iter()
            .copied()
            .max()
            .map_or(0usize, |m| (m as usize) + 1);
        bg_touches_border = (0..bg_count)
            .map(|id| component_touches_border(&bg_labels, width, height, id as i32))
            .collect();
        bg_comp_to_entry = vec![None; bg_count];

        for comp_id in 0..bg_count {
            if bg_touches_border[comp_id] {
                continue;
            }

            let start = find_boundary_start(&bg_labels, width, height, comp_id as i32);
            if let Some(start_idx) = start {
                let contour =
                    trace_component_boundary(&bg_labels, width, height, comp_id as i32, start_idx);
                let contour = approximate(contour, method);

                let parent_hint = find_adjacent_foreground_component(
                    &bg_labels,
                    &fg_labels,
                    width,
                    height,
                    comp_id as i32,
                )
                .and_then(|fg_id| fg_comp_to_entry.get(fg_id as usize).and_then(|x| *x));

                entries.push(ContourEntry {
                    points: contour,
                    kind: ContourKind::Hole,
                    component_id: comp_id as i32,
                    parent_hint,
                });
                bg_comp_to_entry[comp_id] = Some(entries.len() - 1);
            }
        }
    }

    let mut hierarchy = vec![[-1i32; 4]; entries.len()];
    match mode {
        RetrievalMode::External => {
            entries.retain(|e| e.kind == ContourKind::Foreground);
            hierarchy = build_hierarchy_external_like(entries.len(), &[]);
        }
        RetrievalMode::List => {
            hierarchy = build_hierarchy_external_like(entries.len(), &[]);
        }
        RetrievalMode::CComp => {
            let parent: Vec<Option<usize>> = entries
                .iter()
                .map(|e| match e.kind {
                    ContourKind::Foreground => None,
                    ContourKind::Hole => e.parent_hint,
                })
                .collect();
            hierarchy = build_hierarchy_with_parent(&parent);
        }
        RetrievalMode::Tree => {
            let mut parent = vec![None; entries.len()];
            for (idx, e) in entries.iter().enumerate() {
                match e.kind {
                    ContourKind::Hole => {
                        parent[idx] = e.parent_hint;
                    }
                    ContourKind::Foreground => {
                        if let Some(bg_comp_id) = find_adjacent_background_component(
                            &fg_labels,
                            &bg_labels,
                            width,
                            height,
                            e.component_id,
                        ) {
                            let bg_id = bg_comp_id as usize;
                            if bg_id < bg_touches_border.len()
                                && !bg_touches_border[bg_id]
                                && bg_id < bg_comp_to_entry.len()
                            {
                                parent[idx] = bg_comp_to_entry[bg_id];
                            }
                        }
                    }
                }
            }
            hierarchy = build_hierarchy_with_parent(&parent);
        }
    }

    let contours = entries.into_iter().map(|e| e.points).collect();
    Ok(FindContoursResult {
        contours,
        hierarchy,
    })
}

/// Backward-compatible alias for a misspelled API name.
///
/// # Arguments
///
/// * `src` - Binary single-channel image (`0` background, non-zero foreground).
/// * `mode` - Contour retrieval mode.
/// * `method` - Contour approximation method.
///
/// # Returns
///
/// A [`FindContoursResult`] containing contour points and hierarchy.
///
/// # Errors
///
/// Propagates errors from [`find_contours`].
pub fn find_countours<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
) -> Result<FindContoursResult, ImageError> {
    find_contours(src, mode, method)
}

fn label_components(mask: &[bool], width: usize, height: usize) -> Vec<i32> {
    let mut labels = vec![-1i32; width * height];
    let mut next_label = 0i32;
    let mut stack = Vec::<usize>::new();

    for idx in 0..mask.len() {
        if !mask[idx] || labels[idx] >= 0 {
            continue;
        }
        labels[idx] = next_label;
        stack.push(idx);

        while let Some(cur) = stack.pop() {
            let y = cur / width;
            let x = cur % width;

            for &(dx, dy) in &DIRS_8 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if mask[nidx] && labels[nidx] < 0 {
                    labels[nidx] = next_label;
                    stack.push(nidx);
                }
            }
        }
        next_label += 1;
    }
    labels
}

fn find_boundary_start(labels: &[i32], width: usize, height: usize, comp_id: i32) -> Option<usize> {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if labels[idx] == comp_id && is_boundary(labels, width, height, x, y, comp_id) {
                return Some(idx);
            }
        }
    }
    None
}

fn is_boundary(
    labels: &[i32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    comp_id: i32,
) -> bool {
    for &(dx, dy) in &DIRS_8 {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
            return true;
        }
        let nidx = ny as usize * width + nx as usize;
        if labels[nidx] != comp_id {
            return true;
        }
    }
    false
}

fn trace_component_boundary(
    labels: &[i32],
    width: usize,
    height: usize,
    comp_id: i32,
    start_idx: usize,
) -> Contour {
    let start_x = (start_idx % width) as i32;
    let start_y = (start_idx / width) as i32;
    let mut contour = vec![(start_x, start_y)];

    let mut current = (start_x, start_y);
    let mut prev = (start_x - 1, start_y);
    let start_prev = prev;
    let mut safety = 0usize;
    let safety_limit = width * height * 16;

    loop {
        safety += 1;
        if safety > safety_limit {
            break;
        }

        let mut dir_index = 4usize;
        for (i, &(dx, dy)) in DIRS_8.iter().enumerate() {
            if current.0 + dx == prev.0 && current.1 + dy == prev.1 {
                dir_index = i;
                break;
            }
        }

        let mut found_next = None;
        for step in 1..=8 {
            let k = (dir_index + step) % 8;
            let nx = current.0 + DIRS_8[k].0;
            let ny = current.1 + DIRS_8[k].1;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let nidx = ny as usize * width + nx as usize;
            if labels[nidx] == comp_id {
                found_next = Some(((nx, ny), k));
                break;
            }
        }

        let Some((next, k)) = found_next else {
            break;
        };

        prev = (
            current.0 + DIRS_8[(k + 7) % 8].0,
            current.1 + DIRS_8[(k + 7) % 8].1,
        );
        current = next;

        if current == (start_x, start_y) && prev == start_prev {
            break;
        }
        contour.push(current);
    }

    if contour.len() > 1 && contour.first() == contour.last() {
        contour.pop();
    }
    contour
}

fn approximate(mut contour: Contour, mode: ContourApproximationMode) -> Contour {
    if mode == ContourApproximationMode::None || contour.len() <= 2 {
        return contour;
    }

    contour.dedup();
    if contour.len() <= 2 {
        return contour;
    }

    let mut out = Vec::with_capacity(contour.len());
    let n = contour.len();
    for i in 0..n {
        let prev = contour[(i + n - 1) % n];
        let cur = contour[i];
        let next = contour[(i + 1) % n];

        let v1 = (cur.0 - prev.0, cur.1 - prev.1);
        let v2 = (next.0 - cur.0, next.1 - cur.1);
        let cross = v1.0 * v2.1 - v1.1 * v2.0;
        if cross != 0 {
            out.push(cur);
        }
    }

    if out.is_empty() {
        contour
    } else {
        out
    }
}

fn component_touches_border(labels: &[i32], width: usize, height: usize, comp_id: i32) -> bool {
    for x in 0..width {
        if labels[x] == comp_id || labels[(height - 1) * width + x] == comp_id {
            return true;
        }
    }
    for y in 0..height {
        if labels[y * width] == comp_id || labels[y * width + (width - 1)] == comp_id {
            return true;
        }
    }
    false
}

fn find_adjacent_foreground_component(
    bg_labels: &[i32],
    fg_labels: &[i32],
    width: usize,
    height: usize,
    bg_comp_id: i32,
) -> Option<i32> {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if bg_labels[idx] != bg_comp_id {
                continue;
            }
            for &(dx, dy) in &DIRS_8 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if fg_labels[nidx] >= 0 {
                    return Some(fg_labels[nidx]);
                }
            }
        }
    }
    None
}

fn find_adjacent_background_component(
    fg_labels: &[i32],
    bg_labels: &[i32],
    width: usize,
    height: usize,
    fg_comp_id: i32,
) -> Option<i32> {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if fg_labels[idx] != fg_comp_id {
                continue;
            }
            for &(dx, dy) in &DIRS_8 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if bg_labels[nidx] >= 0 {
                    return Some(bg_labels[nidx]);
                }
            }
        }
    }
    None
}

fn build_hierarchy_external_like(num: usize, parent: &[Option<usize>]) -> Vec<[i32; 4]> {
    let mut hierarchy = vec![[-1i32; 4]; num];
    for i in 0..num {
        if i > 0 {
            hierarchy[i][1] = (i - 1) as i32;
        }
        if i + 1 < num {
            hierarchy[i][0] = (i + 1) as i32;
        }
        if let Some(p) = parent.get(i).and_then(|x| *x) {
            hierarchy[i][3] = p as i32;
        }
    }
    hierarchy
}

fn build_hierarchy_with_parent(parent: &[Option<usize>]) -> Vec<[i32; 4]> {
    let n = parent.len();
    let mut hierarchy = vec![[-1i32; 4]; n];
    let mut children = vec![Vec::<usize>::new(); n];
    let mut roots = Vec::<usize>::new();

    for (i, p) in parent.iter().enumerate() {
        if let Some(pid) = p {
            children[*pid].push(i);
            hierarchy[i][3] = *pid as i32;
        } else {
            roots.push(i);
        }
    }

    for siblings in children.iter_mut() {
        siblings.sort_unstable();
        for (k, &idx) in siblings.iter().enumerate() {
            if k > 0 {
                hierarchy[idx][1] = siblings[k - 1] as i32;
            }
            if k + 1 < siblings.len() {
                hierarchy[idx][0] = siblings[k + 1] as i32;
            }
        }
    }

    roots.sort_unstable();
    for (k, &idx) in roots.iter().enumerate() {
        if k > 0 {
            hierarchy[idx][1] = roots[k - 1] as i32;
        }
        if k + 1 < roots.len() {
            hierarchy[idx][0] = roots[k + 1] as i32;
        }
    }

    for i in 0..n {
        if let Some(&first_child) = children[i].first() {
            hierarchy[i][2] = first_child as i32;
        }
    }
    hierarchy
}

#[cfg(test)]
mod tests {
    use super::{find_contours, ContourApproximationMode, RetrievalMode};
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_find_contours_external_rectangle() -> Result<(), ImageError> {
        let data = vec![
            0u8, 0, 0, 0, 0, //
            0, 255, 255, 255, 0, //
            0, 255, 255, 255, 0, //
            0, 255, 255, 255, 0, //
            0, 0, 0, 0, 0,
        ];
        let img = Image::<u8, 1, _>::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            data,
            CpuAllocator,
        )?;
        let out = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        assert_eq!(out.contours.len(), 1);
        assert_eq!(out.hierarchy.len(), 1);
        Ok(())
    }

    #[test]
    fn test_find_contours_hole_tree() -> Result<(), ImageError> {
        let data = vec![
            0u8, 0, 0, 0, 0, 0, 0, //
            0, 255, 255, 255, 255, 255, 0, //
            0, 255, 0, 0, 0, 255, 0, //
            0, 255, 0, 0, 0, 255, 0, //
            0, 255, 255, 255, 255, 255, 0, //
            0, 0, 0, 0, 0, 0, 0,
        ];
        let img = Image::<u8, 1, _>::new(
            ImageSize {
                width: 7,
                height: 6,
            },
            data,
            CpuAllocator,
        )?;
        let out = find_contours(&img, RetrievalMode::Tree, ContourApproximationMode::Simple)?;
        assert!(out.contours.len() >= 2);
        assert_eq!(out.contours.len(), out.hierarchy.len());
        assert!(out.hierarchy.iter().any(|h| h[3] >= 0));
        Ok(())
    }

    #[test]
    fn test_find_contours_simple_vs_none() -> Result<(), ImageError> {
        let data = vec![
            0u8, 0, 0, 0, 0, //
            0, 255, 255, 255, 0, //
            0, 255, 255, 255, 0, //
            0, 255, 255, 255, 0, //
            0, 0, 0, 0, 0,
        ];
        let img = Image::<u8, 1, _>::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            data,
            CpuAllocator,
        )?;
        let out_none = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        let out_simple = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        assert_eq!(out_none.contours.len(), 1);
        assert_eq!(out_simple.contours.len(), 1);
        assert!(out_none.contours[0].len() >= out_simple.contours[0].len());
        Ok(())
    }

    #[test]
    fn test_find_contours_tree_deep_nesting() -> Result<(), ImageError> {
        let data = vec![
            0u8, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 255, 255, 255, 255, 255, 255, 255, 0, //
            0, 255, 0, 0, 0, 0, 0, 255, 0, //
            0, 255, 0, 255, 255, 255, 0, 255, 0, //
            0, 255, 0, 255, 0, 255, 0, 255, 0, //
            0, 255, 0, 255, 255, 255, 0, 255, 0, //
            0, 255, 0, 0, 0, 0, 0, 255, 0, //
            0, 255, 255, 255, 255, 255, 255, 255, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let img = Image::<u8, 1, _>::new(
            ImageSize {
                width: 9,
                height: 9,
            },
            data,
            CpuAllocator,
        )?;

        let out = find_contours(&img, RetrievalMode::Tree, ContourApproximationMode::Simple)?;
        assert!(out.contours.len() >= 3);
        let max_depth = (0..out.hierarchy.len())
            .map(|i| {
                let mut depth = 0usize;
                let mut p = out.hierarchy[i][3];
                while p >= 0 {
                    depth += 1;
                    p = out.hierarchy[p as usize][3];
                }
                depth
            })
            .max()
            .unwrap_or(0);
        assert!(max_depth >= 2);
        Ok(())
    }
}
