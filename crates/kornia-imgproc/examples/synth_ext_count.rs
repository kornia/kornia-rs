//! Try various synthetic patterns and report kornia EXT count vs expected.
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{find_contours, RetrievalMode, ContourApproximationMode};

fn count_ext(data: &[u8], w: usize, h: usize) -> usize {
    let img = Image::<u8, 1, _>::new(ImageSize { width: w, height: h }, data.to_vec(), CpuAllocator).unwrap();
    let r = find_contours(&img, RetrievalMode::External, ContourApproximationMode::Simple).unwrap();
    r.contours.len()
}

fn fill(w: usize, h: usize, rects: &[(usize, usize, usize, usize)]) -> Vec<u8> {
    let mut d = vec![0u8; w * h];
    for &(r0, c0, r1, c1) in rects {
        for r in r0..r1 {
            for c in c0..c1 {
                d[r * w + c] = 1;
            }
        }
    }
    d
}

fn main() {
    let cases: Vec<(&str, Vec<u8>, usize, usize, usize)> = vec![
        // Test "1-pixel-wide row" hypothesis: outer A with a thin
        // single-pixel-wide row, then outer B disjoint to its right.
        ("triangle then disjoint square", {
            let mut d = vec![0u8; 30*8];
            // A: triangle 4 wide at top, narrowing to 1 pixel at row 4.
            //   row 1: cols 1..5
            //   row 2: cols 2..4
            //   row 3: col 2 (1-pixel wide!)
            for c in 1..5 { d[1*30+c] = 1; }
            for c in 2..4 { d[2*30+c] = 1; }
            d[3*30+2] = 1;
            // B: 3x3 square at cols 10..13, rows 2..5
            for r in 2..5 { for c in 10..13 { d[r*30+c] = 1; } }
            d
        }, 30, 8, 2),
        // Spike + disjoint
        ("spike + disjoint", {
            let mut d = vec![0u8; 30*10];
            // A: 1-pixel column at col 2, rows 2..7
            for r in 2..7 { d[r*30+2] = 1; }
            // B: 3x3 square at cols 10..13, rows 4..7
            for r in 4..7 { for c in 10..13 { d[r*30+c] = 1; } }
            d
        }, 30, 10, 2),
        ("2 disjoint squares same row", fill(20, 8, &[(2, 2, 6, 6), (2, 12, 6, 16)]), 20, 8, 2),
        ("2 disjoint squares stacked",  fill(20, 16, &[(2, 2, 6, 6), (10, 2, 14, 6)]), 20, 16, 2),
        ("3 in a row",                  fill(30, 8, &[(2, 2, 6, 6), (2, 12, 6, 16), (2, 22, 6, 26)]), 30, 8, 3),
        ("L-shape",                     fill(8, 8, &[(1, 1, 7, 3), (5, 1, 7, 7)]), 8, 8, 1),
        ("T-shape",                     fill(8, 8, &[(1, 1, 3, 7), (1, 3, 7, 5)]), 8, 8, 1),
        ("U-shape (horseshoe)",         fill(10, 10, &[(1, 1, 8, 3), (1, 7, 8, 9), (6, 1, 8, 9)]), 10, 10, 1),
        ("checker 4x4",                 (0..16).map(|i| (((i/4 + i%4) & 1) as u8)).collect(), 4, 4, 1),
        ("3x3 holed",                   {
            let mut d = vec![1u8; 5*5];
            d[2*5 + 2] = 0;  // single hole
            d
        }, 5, 5, 1),
        ("2 squares with hole in 1",    {
            let mut d = vec![0u8; 20*8];
            for r in 1..7 { for c in 1..7 { d[r*20+c] = 1; } }   // square 1
            for r in 3..5 { for c in 3..5 { d[r*20+c] = 0; } }   // hole in square 1
            for r in 1..7 { for c in 12..18 { d[r*20+c] = 1; } } // square 2
            d
        }, 20, 8, 2),
    ];

    for (name, data, w, h, expected) in cases {
        let actual = count_ext(&data, w, h);
        let ok = if actual == expected { "✓" } else { "❌" };
        println!("{ok} {name}: got {actual}, expected {expected}");
    }
}
