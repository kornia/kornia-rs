use kornia_image::{allocator::ImageAllocator, Image};

pub(crate) fn bicubic_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());

    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let x = u - iu as f32;
    let y = v - iv as f32;

    let get = |row: usize, col: usize| -> f32 {
        let row = row.clamp(0, rows - 1);
        let col = col.clamp(0, cols - 1);

        *image.get_unchecked([row, col, c])
    };

    let mut f = [[0.0; 2]; 2]; // pixel value
    let mut fx = [[0.0; 2]; 2]; // deivative in x dir
    let mut fy = [[0.0; 2]; 2]; // derivative in y dir
    let mut fxy = [[0.0; 2]; 2]; // derivative in xy

    // using central derivative approximative
    for j in 0..2 {
        let rr = iv + j as usize;
        for i in 0..2 {
            let cc = iu + i as usize;
            f[j][i] = get(rr, cc);
            fx[j][i] = (get(rr, cc + 1) - get(rr, cc - 1)) * 0.5;
            fy[j][i] = (get(rr + 1, cc) - get(rr - 1, cc)) * 0.5;
            fxy[j][i] = (get(rr + 1, cc + 1) - get(rr + 1, cc - 1) - get(rr - 1, cc + 1)
                + get(rr - 1, cc - 1))
                * 0.25;
        }
    }

    let matrix_f = [
        [f[0][0], f[0][1], fy[0][0], fy[0][1]],
        [f[1][0], f[1][1], fy[1][0], fy[1][1]],
        [fx[0][0], fx[0][1], fxy[0][0], fxy[0][1]],
        [fx[1][0], fx[1][1], fxy[1][0], fxy[1][1]],
    ];

    let matrix_m = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-3.0, 3.0, -2.0, -1.0],
        [2.0, -2.0, 1.0, 1.0],
    ];

    // A = M * F * M^T
    let mut temp = [[0.0; 4]; 4];
    let mut matrix_a = [[0.0; 4]; 4];

    // M*F = temp
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                temp[i][j] += matrix_m[i][k] * matrix_f[k][j];
            }
        }
    }
    // temp*M^T = A
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                matrix_a[i][j] += temp[i][k] * matrix_m[j][k];
            }
        }
    }

    // [1 x x^2 x^3]*A*[1 y y^2 y^3]^T
    let px = [1.0, x, x * x, x * x * x];
    let py = [1.0, y, y * y, y * y * y];

    let mut res = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            res += px[i] * matrix_a[i][j] * py[j];
        }
    }

    res
}
