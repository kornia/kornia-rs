use std::collections::BinaryHeap;

#[derive(Clone, Debug, PartialEq)]
struct Point {
    coords: [f32; 3],
}

#[derive(PartialEq)]
struct PointWithDistance {
    point: Point,
    distance: f32,
}

impl Eq for PointWithDistance {}
impl Ord for PointWithDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl PartialOrd for PointWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn euclidean_distance(p1: &Point, p2: &Point) -> f32 {
    ((p1.coords[0] - p2.coords[0]).powi(2)
        + (p1.coords[1] - p2.coords[1]).powi(2)
        + (p1.coords[2] - p2.coords[2]).powi(2))
    .sqrt()
}

struct Node {
    points: Point,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    split_dimension: usize,
}

impl Node {
    fn new(mut points: Vec<Point>, depth: usize) -> Option<Box<Node>> {
        if points.is_empty() {
            return None;
        }

        // choose split axis
        let split_dimension = depth % 3;

        // sort points along split dimension
        points.sort_by(|a, b| {
            a.coords[split_dimension]
                .partial_cmp(&b.coords[split_dimension])
                .unwrap()
        });
        let median_index = points.len() / 2;

        // create node with the median point
        let median_point = points[median_index].clone();
        let left_points = points[..median_index].to_vec();
        let right_points = points[median_index + 1..].to_vec();

        Some(Box::new(Node {
            points: median_point,
            left: Node::new(left_points, depth + 1),
            right: Node::new(right_points, depth + 1),
            split_dimension,
        }))
    }

    fn knn_search(&self, query: &Point, k: usize, heap: &mut BinaryHeap<PointWithDistance>) {
        // calculate distance to the point
        let distance = euclidean_distance(&self.points, query);

        // add point to heap
        if heap.len() < k {
            heap.push(PointWithDistance {
                point: self.points.clone(),
                distance,
            });
        } else if let Some(top) = heap.peek() {
            if distance < top.distance {
                heap.pop();
                heap.push(PointWithDistance {
                    point: self.points.clone(),
                    distance,
                });
            }
        }

        // determine which side of the node to search
        let split_distance =
            self.points.coords[self.split_dimension] - query.coords[self.split_dimension];

        // search in the nearest side
        let (nearer, further) = if split_distance < 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };

        if let Some(near_node) = nearer {
            near_node.knn_search(query, k, heap);
        }

        // if there is a further node, search it
        if split_distance.abs() < heap.peek().map_or(f32::INFINITY, |p| p.distance) {
            if let Some(further_node) = further {
                further_node.knn_search(query, k, heap);
            }
        }
    }
}

struct KdeTree {
    root: Option<Box<Node>>,
}

impl KdeTree {
    pub fn new(points: Vec<Point>) -> Self {
        let root = Node::new(points, 0);
        Self { root }
    }

    pub fn knn(&self, query: &Point, k: usize) -> Vec<Point> {
        let mut heap = BinaryHeap::with_capacity(k);
        if let Some(root) = &self.root {
            root.knn_search(query, k, &mut heap);
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|p| p.point)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn() {
        let points = vec![
            Point {
                coords: [0.0, 0.0, 0.0],
            },
            Point {
                coords: [1.0, 0.0, 0.0],
            },
            Point {
                coords: [0.0, 1.0, 0.0],
            },
            Point {
                coords: [0.0, 0.0, 1.0],
            },
        ];
        let tree = KdeTree::new(points);

        let knn = tree.knn(
            &Point {
                coords: [0.0, 0.0, 0.0],
            },
            1,
        );
        println!("res {:?}", knn);
        assert_eq!(
            knn,
            vec![Point {
                coords: [0.0, 0.0, 0.0]
            }]
        );
    }
}
