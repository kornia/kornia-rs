//! Contour finding implementation based on Suzuki and Abe (1985).

use kornia_image::allocator::ImageAllocator;
use kornia_image::Image;

// Public types

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    External,
    List,
    CComp,
    Tree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourApproximationMode {
    None,
    Simple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BorderType {
    Outer,
    Hole,
}

pub type Contour = Vec<[i32; 2]>;
pub type HierarchyEntry = [i32; 4];

pub struct ContoursResult {
    pub contours: Vec<Contour>,
    pub hierarchy: Vec<HierarchyEntry>,
}

// Compile-time tables

// DIR_LUT[((dr+1)*3 + (dc+1))] → direction index 0-7
// Directions: 0=W 1=NW 2=N 3=NE 4=E 5=SE 6=S 7=SW
const DIR_LUT: [usize; 9] = [1, 2, 3, 0, 0, 4, 7, 6, 5];

// Row / column delta per direction (0=W..7=SW)
const DIR_DR: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
const DIR_DC: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

// Reusable executor

pub struct FindContoursExecutor {
    img: Vec<i16>,
}

impl FindContoursExecutor {
    pub fn new() -> Self {
        Self { img: Vec::new() }
    }

    pub fn find_contours<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<ContoursResult, String> {
        find_contours_impl(src, mode, method, &mut self.img)
    }
}

impl Default for FindContoursExecutor {
    fn default() -> Self {
        Self::new()
    }
}


pub fn find_contours<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
) -> Result<ContoursResult, String> {
    let mut img = Vec::new();
    find_contours_impl(src, mode, method, &mut img)
}


fn find_contours_impl<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
    img: &mut Vec<i16>,
) -> Result<ContoursResult, String> {
    let height = src.height();
    let width = src.width();
    let padded_w = width + 2;
    let padded_h = height + 2;
    let padded_n = padded_h * padded_w;

    if img.len() < padded_n {
        img.resize(padded_n, 0);
    }
    let img_slice = &mut img[..padded_n];
    img_slice.fill(0);

    // Binarize src into padded interior: non-zero → 1, zero → 0.
    let src_data = src.as_slice();
    for r in 0..height {
        let src_row = &src_data[r * width..(r + 1) * width];
        let dst_start = (r + 1) * padded_w + 1;
        let dst_row = &mut img_slice[dst_start..dst_start + width];
        for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
            *d = if s != 0 { 1 } else { 0 };
        }
    }

    let pw = padded_w as isize;
    let offsets: [isize; 8] = [-1, -pw - 1, -pw, -pw + 1, 1, pw + 1, pw, pw - 1];

    let mut contours: Vec<Contour> = Vec::new();
    let mut hierarchy: Vec<HierarchyEntry> = Vec::new();
    let mut border_types: Vec<BorderType> = Vec::new();

    let mut nbd: i16 = 1;
    hierarchy.push([-1, -1, -1, -1]);
    border_types.push(BorderType::Outer); // frame sentinel

    let img_ptr = img_slice.as_mut_ptr();

    for r in 1..=height {
        let mut lnbd: i16 = 1;
        let row_base = r * padded_w;
        let mut c = 1usize;

        'col: while c <= width {
            // SWAR zero-skip
            while c + 4 <= width {
                let word = unsafe {
                    (img_ptr.add(row_base + c) as *const u64).read_unaligned()
                };
                if word != 0 { break; }
                c += 4;
                if c > width { break 'col; }
            }

            let idx = row_base + c;
            let pixel = unsafe { *img_ptr.add(idx) };

            if pixel != 0 {
                let left = unsafe { *img_ptr.add(idx - 1) };
                let right = unsafe { *img_ptr.add(idx + 1) };

                let is_outer = pixel == 1 && left == 0;
                let is_hole = pixel >= 1 && right == 0 && !is_outer;

                if is_outer || is_hole {
                    if nbd == i16::MAX {
                        return Err("find_contours: too many borders (nbd i16 overflow)".into());
                    }
                    nbd += 1;

                    let border_type = if is_outer { BorderType::Outer } else { BorderType::Hole };
                    let parent = determine_parent(lnbd as i32, border_type, &hierarchy, &border_types);
                    let start_dir: usize = if is_outer { 0 } else { 4 };

                    let contour = trace_border(
                        img_ptr,
                        padded_w,
                        idx,
                        r as i32,
                        c as i32,
                        start_dir,
                        nbd,
                        method,
                        &offsets,
                    );

                    let hier_entry = update_hierarchy(&mut hierarchy, nbd as usize, parent);
                    hierarchy.push(hier_entry);
                    border_types.push(border_type);
                    contours.push(contour);
                }

                let abs_pixel = pixel.abs();
                if abs_pixel > 1 {
                    lnbd = abs_pixel;
                }
            }
            c += 1;
        }
    }

    Ok(filter_by_mode(contours, hierarchy, border_types, mode))
}

// Border tracer

#[inline(always)]
fn trace_border(
    img: *mut i16,
    padded_w: usize,
    start_idx: usize,
    start_row: i32,
    start_col: i32,
    start_dir: usize,
    nbd: i16,
    method: ContourApproximationMode,
    offsets: &[isize; 8],
) -> Contour {
    let mut first_nb_idx = 0usize;
    let mut first_nb_dir = 0usize;
    let mut found = false;

    for k in 0..8usize {
        let d = (start_dir + k) & 7;
        let nb = unsafe { (start_idx as isize + *offsets.get_unchecked(d)) as usize };
        if unsafe { *img.add(nb) } != 0 {
            first_nb_idx = nb;
            first_nb_dir = d;
            found = true;
            break;
        }
    }

    if !found {
        unsafe { *img.add(start_idx) = -nbd };
        return vec![[start_col - 1, start_row - 1]];
    }

    let mut contour = Vec::with_capacity(64);
    contour.push([start_col - 1, start_row - 1]);

    let mut i2_idx = first_nb_idx;
    let mut i2_row = start_row + DIR_DR[first_nb_dir];
    let mut i2_col = start_col + DIR_DC[first_nb_dir];
    let mut dir_in = first_nb_dir;

    loop {
        let cur = unsafe { *img.add(i2_idx) };
        let left_nb = unsafe { *img.add(i2_idx - 1) };
        let right_nb = unsafe { *img.add(i2_idx + 1) };

        if left_nb == 0 && cur == 1 {
            unsafe { *img.add(i2_idx) = nbd };
        } else if right_nb == 0 && cur > 0 {
            unsafe { *img.add(i2_idx) = -nbd };
        }

        let mut scan_dir = (dir_in + 5) & 7;
        let mut i3_idx = 0usize;
        let mut i3_row = 0i32;
        let mut i3_col = 0i32;
        let mut dir_out = 0usize;

        for _ in 0..8usize {
            let nb = unsafe { (i2_idx as isize + *offsets.get_unchecked(scan_dir)) as usize };
            if unsafe { *img.add(nb) } != 0 {
                i3_idx = nb;
                i3_row = i2_row + DIR_DR[scan_dir];
                i3_col = i2_col + DIR_DC[scan_dir];
                dir_out = scan_dir;
                break;
            }
            scan_dir = (scan_dir + 1) & 7;
        }

        if i2_idx == start_idx && i3_idx == first_nb_idx {
            break;
        }

        match method {
            ContourApproximationMode::None => {
                contour.push([i2_col - 1, i2_row - 1]);
            }
            ContourApproximationMode::Simple => {
                if dir_in != dir_out {
                    contour.push([i2_col - 1, i2_row - 1]);
                }
            }
        }

        i2_idx = i3_idx;
        i2_row = i3_row;
        i2_col = i3_col;
        dir_in = dir_out;
    }

    if method == ContourApproximationMode::Simple && i2_idx != start_idx {
        let dr = start_row - i2_row;
        let dc = start_col - i2_col;
        let dir_out = DIR_LUT[((dr + 1) * 3 + (dc + 1)) as usize];
        let pt = [i2_col - 1, i2_row - 1];
        if dir_in != dir_out && contour.last() != Some(&pt) {
            contour.push(pt);
        }
    }

    if unsafe { *img.add(start_idx) } == 1 {
        unsafe { *img.add(start_idx) = nbd };
    }

    contour
}

// Hierarchy helpers

#[inline(always)]
fn determine_parent(
    lnbd: i32,
    border_type: BorderType,
    hierarchy: &[HierarchyEntry],
    border_types: &[BorderType],
) -> i32 {
    let lnbd_idx = (lnbd - 1) as usize;
    if lnbd_idx >= hierarchy.len() {
        return -1;
    }
    let lnbd_is_hole = matches!(border_types[lnbd_idx], BorderType::Hole);
    match border_type {
        BorderType::Outer => hierarchy[lnbd_idx][3],
        BorderType::Hole => if lnbd_is_hole { hierarchy[lnbd_idx][3] } else { lnbd }
    }
}

#[inline(always)]
fn update_hierarchy(
    hierarchy: &mut Vec<HierarchyEntry>,
    nbd: usize,
    parent: i32,
) -> HierarchyEntry {
    let mut entry = [-1i32, -1, -1, parent];
    if parent >= 0 {
        let pidx = (parent - 1) as usize;
        if pidx < hierarchy.len() {
            if hierarchy[pidx][2] == -1 {
                hierarchy[pidx][2] = nbd as i32;
            } else {
                let mut sib = hierarchy[pidx][2] as usize;
                while hierarchy[sib - 1][0] != -1 {
                    sib = hierarchy[sib - 1][0] as usize;
                }
                hierarchy[sib - 1][0] = nbd as i32;
                entry[1] = sib as i32;
            }
        }
    }
    entry
}

// Mode filter

fn filter_by_mode(
    contours: Vec<Contour>,
    hierarchy: Vec<HierarchyEntry>,
    border_types: Vec<BorderType>,
    mode: RetrievalMode,
) -> ContoursResult {
    match mode {
        RetrievalMode::External => {
            let mut fc = Vec::new();
            let mut fh = Vec::new();
            for (i, h) in hierarchy.iter().enumerate().skip(1) {
                if matches!(border_types[i], BorderType::Outer) && h[3] <= 1 {
                    fc.push(contours[i - 1].clone());
                    fh.push([-1i32, -1, -1, -1]);
                }
            }
            ContoursResult { contours: fc, hierarchy: fh }
        }
        RetrievalMode::List => {
            let fh = vec![[-1i32, -1, -1, -1]; contours.len()];
            ContoursResult { contours, hierarchy: fh }
        }
        _ => ContoursResult {
            contours,
            hierarchy: hierarchy[1..].to_vec(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_tensor::allocator::CpuAllocator;
    use kornia_tensor::Tensor3;

    fn make_img(w: usize, h: usize, data: Vec<u8>) -> Image<u8, 1, CpuAllocator> {
        Image(Tensor3::from_shape_vec([h, w, 1], data, CpuAllocator).expect("tensor"))
    }

    #[test]
    fn test_simple_square_no_approx() {
        #[rustfmt::skip]
        let img = make_img(5, 5, vec![
            0,0,0,0,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
        ]);
        let r = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None).unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 8);
    }

    #[test]
    fn test_simple_square_simple_approx() {
        #[rustfmt::skip]
        let img = make_img(5, 5, vec![
            0,0,0,0,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
        ]);
        let r = find_contours(&img, RetrievalMode::External, ContourApproximationMode::Simple).unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 4);
    }

    #[test]
    fn test_hollow_square_external_vs_list() {
        #[rustfmt::skip]
        let img = make_img(6, 6, vec![
            0,0,0,0,0,0,
            0,1,1,1,1,0,
            0,1,0,0,1,0,
            0,1,0,0,1,0,
            0,1,1,1,1,0,
            0,0,0,0,0,0,
        ]);
        let ext = find_contours(&img, RetrievalMode::External, ContourApproximationMode::Simple).unwrap();
        assert_eq!(ext.contours.len(), 1);
        let list = find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple).unwrap();
        assert_eq!(list.contours.len(), 2);
    }

    #[test]
    fn test_isolated_pixel() {
        let img = make_img(3, 3, vec![0,0,0, 0,1,0, 0,0,0]);
        let r = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None).unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 1);
    }

    #[test]
    fn test_u_shape() {
        #[rustfmt::skip]
        let img = make_img(5, 5, vec![
            0,0,0,0,0,
            0,1,0,1,0,
            0,1,0,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
        ]);
        let r = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None).unwrap();
        assert_eq!(r.contours.len(), 1);
        assert!(r.contours[0].len() >= 7);
    }

    #[test]
    fn test_no_empty_contour_simple_approx() {
        #[rustfmt::skip]
        let img = make_img(5, 5, vec![
            0,0,0,0,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
            0,0,0,0,0,
        ]);
        let r = find_contours(&img, RetrievalMode::External, ContourApproximationMode::Simple).unwrap();
        assert!(!r.contours.is_empty());
        assert!(!r.contours[0].is_empty());
    }

    #[test]
    fn test_many_contours_no_overflow() {
        const W: usize = 21;
        const H: usize = 21;
        let mut data = vec![0u8; W * H];
        for i in 0..63usize {
            let row = (i / 7) * 2 + 1;
            let col = (i % 7) * 2 + 1;
            data[row * W + col] = 1;
        }
        let img = make_img(W, H, data);
        let result = find_contours(&img, RetrievalMode::List, ContourApproximationMode::None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().contours.len(), 63);
    }

    #[test]
    fn test_executor_reuse_identical() {
        #[rustfmt::skip]
        let data = vec![
            0,0,0,0,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
        ];
        let mut exec = FindContoursExecutor::new();
        let i1 = make_img(5, 5, data.clone());
        let i2 = make_img(5, 5, data);
        let r1 = exec.find_contours(&i1, RetrievalMode::External, ContourApproximationMode::None).unwrap();
        let r2 = exec.find_contours(&i2, RetrievalMode::External, ContourApproximationMode::None).unwrap();
        assert_eq!(r1.contours, r2.contours);
    }
}