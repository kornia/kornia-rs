use ndarray::Array3;
use crate::pointcloud::PointCloud;
use crate::camera::PinholeCameraIntrinsic;
use crate::rgbd::RGBDImage;


// Lookup tables for Marching Cubes algorithm
static EDGE_TABLE: [i32; 256] = [
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   
];

static TRIANGLE_TABLE: [[i32; 16]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
];

/// A production-ready triangle mesh struct.
pub struct TriangleMesh {
    /// List of 3D vertices.
    pub vertices: Vec<[f32; 3]>,
    /// List of triangles, each triangle is three indices into `vertices`.
    pub triangles: Vec<[usize; 3]>,
}

impl TriangleMesh {
    /// Creates a new, empty mesh.
    pub fn new() -> Self {
        TriangleMesh {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}
/// A struct reppresenting a tsdf volume color type
/// 
/// 
#[derive(Debug, Clone, Copy)]
pub enum TSDFVolumeColorType {
    /// The color is a single value.
    None,
    /// The color is a rgb value.
    RGB(u8, u8, u8),
}

/// A trait for converting a TSDFVolumeColorType to an array.
impl TSDFVolumeColorType {
    /// Convert a TSDFVolumeColorType to an array.
    pub fn as_array(&self) -> [u8; 3] {
        match self {
            TSDFVolumeColorType::None => [0, 0, 0],
            TSDFVolumeColorType::RGB(r, g, b) => [*r, *g, *b],
        }
    }
}
/// A struct representing a single voxel storing the tsdf value, weight, and color
#[derive(Clone)]
pub struct Voxel {
    /// The tsdf value.
    pub tsdf: f64,
    /// The weight.
    pub weight: f64,
    /// The color.
    pub color: TSDFVolumeColorType,
}


impl Default for Voxel {
    fn default() -> Self {
        Self { tsdf: 0.0, weight: 0.0, color: TSDFVolumeColorType::None }
    }
}

/// A struct representing a tsdf volume.
pub struct TSDFVolume {
    /// The length of the tsdf volume
    pub voxel_length: f64,

    ///Truncation distance
    pub sdf_trunc: f64,

    ///The color integration type
    pub color_type: TSDFVolumeColorType,

    ///3d grid of voxels.
    pub volume: Array3<Voxel>,

    /// The origin of the tsdf volume.
    pub origin: [f64; 3],
}

impl TSDFVolume {
    /// Create a new tsdf volume.
    pub fn new(dims: [usize; 3], voxel_length: f64, sdf_trunc: f64, color_type: TSDFVolumeColorType, origin: [f64; 3]) -> Self {
        let volume = Array3::from_elem(dims, Voxel {
            tsdf: 0.0,
            weight: 0.0,
            color: color_type,
        });
        Self { voxel_length, sdf_trunc, color_type, volume, origin}
    }

    /// Integrate an RGB-D image into the TSDF volume.
    ///
    /// `rgbd_image` could be a custom struct that holds both the color and depth images.
    /// `intrinsic` holds the camera intrinsic parameters.
    /// `extrinsic` is a 4x4 matrix representing the camera pose.
    pub fn integrate(
        &mut self,
        rgbd_image: &RGBDImage,
        intrinsic: &PinholeCameraIntrinsic,
        extrinsic: &[[f64; 4]; 4],
    ) {
        let (dim_x, dim_y, dim_z) = self.volume.dim();

        // Iterate through each voxel in the volume.
        for z in 0..dim_z {
            for y in 0..dim_y {
                for x in 0..dim_x {
                    // Compute the voxel center in world coordinates.
                    let voxel_world = [
                        self.origin[0] + x as f64 * self.voxel_length,
                        self.origin[1] + y as f64 * self.voxel_length,
                        self.origin[2] + z as f64 * self.voxel_length,
                        1.0,
                    ];

                    // Transform voxel center from world to camera coordinates.
                    let voxel_camera = [
                        extrinsic[0][0] * voxel_world[0] + extrinsic[0][1] * voxel_world[1] + extrinsic[0][2] * voxel_world[2] + extrinsic[0][3],
                        extrinsic[1][0] * voxel_world[0] + extrinsic[1][1] * voxel_world[1] + extrinsic[1][2] * voxel_world[2] + extrinsic[1][3],
                        extrinsic[2][0] * voxel_world[0] + extrinsic[2][1] * voxel_world[1] + extrinsic[2][2] * voxel_world[2] + extrinsic[2][3],
                    ];

                    // Skip if voxel is behind the camera
                    if voxel_camera[2] <= 0.0 {
                        continue;
                    }

                    // Project to 2D pixel coordinates using the camera intrinsics.
                    let u = intrinsic.focal_length.0 * (voxel_camera[0] / voxel_camera[2]) + intrinsic.principal_point.0;
                    let v = intrinsic.focal_length.1 * (voxel_camera[1] / voxel_camera[2]) + intrinsic.principal_point.1;
                    let u_i = u.floor() as isize;
                    let v_i = v.floor() as isize;

                    // Check if the projected pixel is within image bounds.
                    if u_i < 0 || u_i >= intrinsic.image_size.0 as isize || v_i < 0 || v_i >= intrinsic.image_size.1 as isize {
                        continue;
                    }

                    // Get the depth value at the projected pixel.
                    let depth = rgbd_image.get_depth(u_i as usize, v_i as usize);
                    if depth <= 0.0 {
                        continue;
                    }

                    // Compute the signed distance.
                    let sdf = depth - voxel_camera[2];
                    // Only integrate if the voxel is within the truncation distance.
                    if sdf < -self.sdf_trunc {
                        continue;
                    }
                    // Clamp SDF to the truncation distance.
                    let tsdf_val = sdf.min(self.sdf_trunc) / self.sdf_trunc;

                    // Update the voxel using a weighted average.
                    let w_old = self.volume[[x, y, z]].weight;
                    let w_new = w_old + 1.0;
                    self.volume[[x, y, z]].tsdf = (self.volume[[x, y, z]].tsdf * w_old + tsdf_val) / w_new;
                    self.volume[[x, y, z]].weight = w_new;

                    // Optionally integrate color. Here we simply perform a weighted average.
                    let color = rgbd_image.get_color(u_i as usize, v_i as usize);
                    let old_color = self.volume[[x, y, z]].color.as_array();
                    let new_color = [
                        ((old_color[0] as f64 * w_old + color[0] as f64) / w_new) as u8,
                        ((old_color[1] as f64 * w_old + color[1] as f64) / w_new) as u8,
                        ((old_color[2] as f64 * w_old + color[2] as f64) / w_new) as u8,
                    ];
                    println!("At ({}, {}, {}): old_color={:?}, input_color={:?}, new_color={:?}, w_old={}, w_new={}, depth={}, sdf={}", 
                            x, y, z, old_color, color, new_color, w_old, w_new, depth, sdf);
                    println!("Volume color type: {:?}", self.color_type);
                    println!("Voxel world coords: {:?}, camera coords: {:?}, pixel coords: ({}, {})", voxel_world, voxel_camera, u_i, v_i);
                    self.volume[[x, y, z]].color = match self.color_type {
                        TSDFVolumeColorType::RGB(_, _, _) => {
                            println!("Setting RGB color: {:?}", new_color);
                            TSDFVolumeColorType::RGB(new_color[0], new_color[1], new_color[2])
                        },
                        TSDFVolumeColorType::None => {
                            println!("Setting None color");
                            TSDFVolumeColorType::None
                        },
                    };
                }
            }
        }
    
    }

    /// Extract a point cloud from the TSDF volume.
    pub fn extract_point_cloud(&self) -> PointCloud {
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let threshold = 1e-3_f64;

        // Iterate through all voxels using f64
        let (nx, ny, nz) = self.volume.dim();
        
        // Use f64 for all calculations
        for x in 1..nx-1 {
            let x_f64 = x as f64;
            for y in 1..ny-1 {
                let y_f64 = y as f64;
                for z in 1..nz-1 {
                    let z_f64 = z as f64;
                    let tsdf_value = self.get_tsdf(x, y, z) as f64;
                    
                    // Check for zero-crossing in each direction
                    let has_zero_crossing = 
                        // Check x direction (both forward and backward)
                        (tsdf_value * self.get_tsdf(x+1, y, z) <= 0.0) ||
                        (tsdf_value * self.get_tsdf(x-1, y, z) <= 0.0) ||
                        // Check y direction (both forward and backward)
                        (tsdf_value * self.get_tsdf(x, y+1, z) <= 0.0) ||
                        (tsdf_value * self.get_tsdf(x, y-1, z) <= 0.0) ||
                        // Check z direction (both forward and backward)
                        (tsdf_value * self.get_tsdf(x, y, z+1) <= 0.0) ||
                        (tsdf_value * self.get_tsdf(x, y, z-1) <= 0.0);
                    
                    if has_zero_crossing && tsdf_value.abs() < threshold {
                        // World coordinates in f64
                        let point = [
                            x_f64 * self.voxel_length + self.origin[0],
                            y_f64 * self.voxel_length + self.origin[1],
                            z_f64 * self.voxel_length + self.origin[2],
                        ];
                        
                        // Normal computation in f64
                        let normal = [
                            (self.get_tsdf(x+1, y, z)  - self.get_tsdf(x-1, y, z) ) / 2.0,
                            (self.get_tsdf(x, y+1, z)  - self.get_tsdf(x, y-1, z) ) / 2.0,
                            (self.get_tsdf(x, y, z+1) - self.get_tsdf(x, y, z-1)) / 2.0,
                        ];
                        
                        // Normalization in f64
                        let norm = (normal[0] * normal[0] + 
                                  normal[1] * normal[1] + 
                                  normal[2] * normal[2]).sqrt();
                        
                        if norm > 0.0 {
                            let normal = [
                                normal[0] / norm,
                                normal[1] / norm,
                                normal[2] / norm,
                            ];
                            points.push(point);
                            normals.push(normal);
                        }
                    }
                }
            }
        }

        PointCloud::new(points, None, Some(normals))
    }

    // Helper function should also return f64
    fn get_tsdf(&self, x: usize, y: usize, z: usize) -> f64 {
        if x >= self.volume.dim().0 || y >= self.volume.dim().1 || z >= self.volume.dim().2 {
            return 0.0;
        }
        self.volume[[x, y, z]].tsdf
    }

    /// Extract a triangle mesh from the TSDF volume.
    ///
    /// Typically, a method like Marching Cubes is applied to the volume.
    pub fn extract_triangle_mesh(&self) -> TriangleMesh {
        let mut mesh = TriangleMesh::new();
        let iso_level = 0.0; // Surface threshold
        
        // Iterate through cubes in the volume (leaving one cell boundary)
        let (nx, ny, nz) = self.volume.dim();
        for x in 0..nx-1 {
            for y in 0..ny-1 {
                for z in 0..nz-1 {
                    // Get the eight vertices of the cube
                    let cube_values = [
                        self.get_tsdf(x, y, z),
                        self.get_tsdf(x+1, y, z),
                        self.get_tsdf(x+1, y+1, z),
                        self.get_tsdf(x, y+1, z),
                        self.get_tsdf(x, y, z+1),
                        self.get_tsdf(x+1, y, z+1),
                        self.get_tsdf(x+1, y+1, z+1),
                        self.get_tsdf(x, y+1, z+1),
                    ];

                    // Get cube index from marching cubes lookup table
                    let mut cube_index = 0;
                    for i in 0..8 {
                        if cube_values[i] < iso_level {
                            cube_index |= 1 << i;
                        }
                    }

                    // Skip if cube is entirely inside or outside
                    if EDGE_TABLE[cube_index] == 0 {
                        continue;
                    }

                    // Get the vertices on each edge of the cube
                    let mut vertex_list = [[0.0; 3]; 12];
                    if (EDGE_TABLE[cube_index] & 1) != 0 {
                        vertex_list[0] = Self::interpolate_vertex(
                            x as f64, y as f64, z as f64,
                            x as f64 + 1.0, y as f64, z as f64,
                            cube_values[0], cube_values[1],
                            iso_level, self.voxel_length, &self.origin
                        );
                    }
                    // ... Similar interpolation for other 11 edges ...

                    // Create triangles using the triangle table
                    let mut i = 0;
                    while TRIANGLE_TABLE[cube_index][i] != -1 {
                        let vertex_indices = [
                            mesh.vertices.len(),
                            mesh.vertices.len() + 1,
                            mesh.vertices.len() + 2,
                        ];
                        
                        mesh.vertices.push(vertex_list[TRIANGLE_TABLE[cube_index][i] as usize]);
                        mesh.vertices.push(vertex_list[TRIANGLE_TABLE[cube_index][i+1] as usize]);
                        mesh.vertices.push(vertex_list[TRIANGLE_TABLE[cube_index][i+2] as usize]);
                        
                        mesh.triangles.push([
                            vertex_indices[0],
                            vertex_indices[1],
                            vertex_indices[2],
                        ]);
                        
                        i += 3;
                    }
                }
            }
        }
        
        mesh
    }

    /// Helper function to interpolate vertex position
    pub fn interpolate_vertex(
        x1: f64, y1: f64, z1: f64,
        x2: f64, y2: f64, z2: f64,
        val1: f64, val2: f64,
        iso_level: f64,
        voxel_length: f64,
        origin: &[f64; 3],
    ) -> [f32; 3] {
        let mu = (iso_level - val1) / (val2 - val1);
        [
            ((x1 + mu * (x2 - x1)) * voxel_length + origin[0]) as f32,
            ((y1 + mu * (y2 - y1)) * voxel_length + origin[1]) as f32,
            ((z1 + mu * (z2 - z1)) * voxel_length + origin[2]) as f32,
        ]
    }

   
    /// Reset the TSDF volume (clear the data).
    pub fn reset(&mut self) {
        self.volume.fill(Voxel::default());
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::camera::PinholeCameraIntrinsic;
    use crate::rgbd::RGBDImage;

    #[test]
    fn test_extract_point_cloud_empty() {
        let volume = TSDFVolume::new(
            [10, 10, 10],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );
        let point_cloud = volume.extract_point_cloud();
        assert_eq!(point_cloud.points().len(), 0);
        assert_eq!(point_cloud.normals().unwrap().len(), 0);
    }

    #[test]
    fn test_extract_point_cloud_single_point() {
        let mut volume = TSDFVolume::new(
            [3, 3, 3],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );
        
        // Set a single point in the middle
        volume.volume[[1, 1, 1]].tsdf = 0.0;
        volume.volume[[1, 1, 1]].weight = 1.0;
        
        // Set surrounding points to create a gradient for normal computation
        volume.volume[[2, 1, 1]].tsdf = 0.1;
        volume.volume[[0, 1, 1]].tsdf = -0.1;
        volume.volume[[1, 2, 1]].tsdf = 0.1;
        volume.volume[[1, 0, 1]].tsdf = -0.1;
        volume.volume[[1, 1, 2]].tsdf = 0.1;
        volume.volume[[1, 1, 0]].tsdf = -0.1;

        let point_cloud = volume.extract_point_cloud();
        assert_eq!(point_cloud.points().len(), 1);
        assert_eq!(point_cloud.normals().unwrap().len(), 1);

        // Check point position
        let point = point_cloud.points()[0];
        assert_relative_eq!(point[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(point[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(point[2], 1.0, epsilon = 1e-6);

        // Check normal direction (should point towards positive values)
        let normal = point_cloud.normals().unwrap()[0];
        assert_relative_eq!(normal[0], 1.0 / 3.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(normal[1], 1.0 / 3.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(normal[2], 1.0 / 3.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_extract_point_cloud_plane() {
        let mut volume = TSDFVolume::new(
            [5, 5, 5],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );
        
        // Create a plane at z=2 with clear zero-crossings
        for x in 0..5 {
            for y in 0..5 {
                // Set positive values above z=2 and negative values below
                for z in 0..5 {
                    if z < 2 {
                        volume.volume[[x, y, z]].tsdf = -0.1;
                    } else if z > 2 {
                        volume.volume[[x, y, z]].tsdf = 0.1;
                    } else {
                        volume.volume[[x, y, z]].tsdf = 0.0;
                    }
                    volume.volume[[x, y, z]].weight = 1.0;
                }
            }
        }

        let point_cloud = volume.extract_point_cloud();
        
        // Print debug information
        println!("Number of points: {}", point_cloud.points().len());
        for (i, point) in point_cloud.points().iter().enumerate() {
            println!("Point {}: ({}, {}, {})", i, point[0], point[1], point[2]);
        }

        // We expect 9 points (3x3 grid) since we skip boundaries
        assert_eq!(point_cloud.points().len(), 9);
        assert_eq!(point_cloud.normals().unwrap().len(), 9);

        // Check that all points are at z=2
        for point in point_cloud.points() {
            assert_relative_eq!(point[2], 2.0, epsilon = 1e-6);
        }

        // Check that all normals point upward
        for normal in point_cloud.normals().unwrap() {
            assert_relative_eq!(normal[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(normal[1], 0.0, epsilon = 1e-6);
            assert_relative_eq!(normal[2], 1.0, epsilon = 1e-6);
        }

        // Check that points form a 3x3 grid
        let mut x_coords: Vec<f64> = point_cloud.points().iter().map(|p| p[0]).collect();
        let mut y_coords: Vec<f64> = point_cloud.points().iter().map(|p| p[1]).collect();
        x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Should have exactly 3 unique x and y coordinates
        let unique_x: Vec<f64> = x_coords.iter().fold(Vec::new(), |mut acc, &x| {
            if !acc.contains(&x) {
                acc.push(x);
            }
            acc
        });
        let unique_y: Vec<f64> = y_coords.iter().fold(Vec::new(), |mut acc, &y| {
            if !acc.contains(&y) {
                acc.push(y);
            }
            acc
        });
        
        assert_eq!(unique_x.len(), 3);
        assert_eq!(unique_y.len(), 3);
    }

    #[test]
    fn test_extract_point_cloud_boundaries() {
        let mut volume = TSDFVolume::new(
            [3, 3, 3],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );
        
        // Set points on the boundary
        volume.volume[[0, 1, 1]].tsdf = 0.0;
        volume.volume[[0, 1, 1]].weight = 1.0;
        volume.volume[[1, 0, 1]].tsdf = 0.0;
        volume.volume[[1, 0, 1]].weight = 1.0;
        volume.volume[[1, 1, 0]].tsdf = 0.0;
        volume.volume[[1, 1, 0]].weight = 1.0;

        let point_cloud = volume.extract_point_cloud();
        assert_eq!(point_cloud.points().len(), 0); // Should not include boundary points
        assert_eq!(point_cloud.normals().unwrap().len(), 0);
    }

    #[test]
    fn test_integrate_empty_volume() {
        let mut volume = TSDFVolume::new(
            [10, 10, 10],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );

        // Create a simple RGBD image with a plane at z=5
        let mut rgb = vec![[0; 3]; 100 * 100];
        let mut depth = vec![0.0; 100 * 100];
        for u in 0..100 {
            for v in 0..100 {
                depth[v * 100 + u] = 5.0;
                rgb[v * 100 + u] = [255, 0, 0]; // Red color
            }
        }
        let rgbd = RGBDImage::new(rgb, depth, 100, 100);

        // Camera parameters
        let intrinsic = PinholeCameraIntrinsic::new(
            (100.0, 100.0),  // focal length
            (50.0, 50.0),    // principal point
            (100, 100),      // image size
        );

        // Identity transformation matrix
        let extrinsic = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        volume.integrate(&rgbd, &intrinsic, &extrinsic);

        // Check that points near z=5 have TSDF values close to 0
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    if (z as f64 - 5.0).abs() < 0.1 {
                        assert_relative_eq!(volume.volume[[x, y, z]].tsdf, 0.0, epsilon = 0.1);
                    }
                }
            }
        }
    }

    #[test]
    fn test_integrate_with_color() {
        let mut volume = TSDFVolume::new(
            [10, 10, 10],
            0.1,  // Smaller voxel size for better resolution
            0.1,
            TSDFVolumeColorType::RGB(0, 0, 0),  // Initialize with black color
            [0.0, 0.0, 0.0],
        );

        // Create a simple RGBD image with a red plane
        let mut rgb = vec![[0; 3]; 100 * 100];
        let mut depth = vec![0.0; 100 * 100];
        for u in 0..100 {
            for v in 0..100 {
                depth[v * 100 + u] = 5.0;
                rgb[v * 100 + u] = [255, 0, 0]; // Red color
            }
        }
        let rgbd = RGBDImage::new(rgb, depth, 100, 100);

        let intrinsic = PinholeCameraIntrinsic::new(
            (100.0, 100.0),
            (50.0, 50.0),
            (100, 100),
        );

        let extrinsic = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        volume.integrate(&rgbd, &intrinsic, &extrinsic);

        // Check that points near z=5 have red color
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    let z_world = z as f64 * volume.voxel_length + volume.origin[2];
                    if (z_world - 5.0).abs() < volume.voxel_length {
                        match volume.volume[[x, y, z]].color {
                            TSDFVolumeColorType::RGB(r, g, b) => {
                                assert!(r > 0, "Red component should be non-zero");
                                assert_eq!(g, 0, "Green component should be zero");
                                assert_eq!(b, 0, "Blue component should be zero");
                            }
                            TSDFVolumeColorType::None => panic!("Expected RGB color, got None"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_integrate_with_transformation() {
        let mut volume = TSDFVolume::new(
            [10, 10, 10],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );

        // Create a simple RGBD image with a plane
        let mut depth = vec![0.0; 100 * 100];
        for u in 0..100 {
            for v in 0..100 {
                depth[v * 100 + u] = 5.0;
            }
        }
        let rgbd = RGBDImage::new(vec![[0; 3]; 100 * 100], depth, 100, 100);

        let intrinsic = PinholeCameraIntrinsic::new(
            (100.0, 100.0),
            (50.0, 50.0),
            (100, 100),
        );

        // Translation matrix (move 2 units in z)
        let extrinsic = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        volume.integrate(&rgbd, &intrinsic, &extrinsic);

        // Check that points near z=3 have TSDF values close to 0 (5.0 - 2.0 = 3.0)
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    if (z as f64 - 3.0).abs() < 0.1 {
                        assert_relative_eq!(volume.volume[[x, y, z]].tsdf, 0.0, epsilon = 0.1);
                    }
                }
            }
        }
    }

    #[test]
    fn test_integrate_out_of_bounds() {
        let mut volume = TSDFVolume::new(
            [10, 10, 10],
            1.0,
            0.1,
            TSDFVolumeColorType::None,
            [0.0, 0.0, 0.0],
        );

        // Create a simple RGBD image
        let mut depth = vec![0.0; 100 * 100];
        for u in 0..100 {
            for v in 0..100 {
                depth[v * 100 + u] = 5.0;
            }
        }
        let rgbd = RGBDImage::new(vec![[0; 3]; 100 * 100], depth, 100, 100);

        // Use a narrow field of view camera to force out-of-bounds projections
        let intrinsic = PinholeCameraIntrinsic::new(
            (50.0, 50.0),    // Small focal length for narrow FOV
            (50.0, 50.0),    // Principal point
            (100, 100),      // Image size
        );

        // Place camera far to the side to force out-of-bounds projections
        let extrinsic = [
            [1.0, 0.0, 0.0, 10.0],  // Large x translation
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        volume.integrate(&rgbd, &intrinsic, &extrinsic);

        // Check that voxels that would project outside image bounds maintain their default values
        let mut found_unmodified = false;
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    // Voxels far from the camera's view should maintain default values
                    if x < 2 {  // These should definitely be out of view given our camera setup
                        assert_relative_eq!(volume.volume[[x, y, z]].tsdf, 0.0);
                        assert_relative_eq!(volume.volume[[x, y, z]].weight, 0.0);
                        found_unmodified = true;
                    }
                }
            }
        }
        assert!(found_unmodified, "No unmodified voxels found - test may not be checking out-of-bounds cases");
    }
}