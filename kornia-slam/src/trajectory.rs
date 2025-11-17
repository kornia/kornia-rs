use glam::Vec2;

/// Trajectory representation with waypoints
#[derive(Debug, Clone)]
pub struct Trajectory {
    waypoints: Vec<Vec2>,
}

impl Trajectory {
    /// Create a new trajectory from waypoints
    pub fn new(waypoints: Vec<Vec2>) -> Self {
        Self { waypoints }
    }

    /// Get the number of waypoints
    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    /// Check if trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Get waypoint at index
    pub fn get_waypoint(&self, index: usize) -> Option<Vec2> {
        self.waypoints.get(index).copied()
    }

    /// Find the closest point on the trajectory to the given position
    pub fn find_closest_point(&self, position: Vec2) -> Option<(usize, f32)> {
        if self.waypoints.is_empty() {
            return None;
        }

        let mut min_dist = f32::MAX;
        let mut closest_idx = 0;

        for (i, wp) in self.waypoints.iter().enumerate() {
            let dist = position.distance(*wp);
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        Some((closest_idx, min_dist))
    }

    /// Get a reference point ahead of the current position
    /// Returns (reference_point, target_heading)
    pub fn get_reference_point(&self, position: Vec2, look_ahead: f32) -> Option<(Vec2, f32)> {
        if self.waypoints.len() < 2 {
            return self.waypoints.first().map(|&wp| (wp, 0.0));
        }

        // Find closest segment
        let (closest_idx, _) = self.find_closest_point(position)?;

        // Start from closest waypoint and find point at look_ahead distance
        let mut accumulated_dist = 0.0;
        let mut current_idx = closest_idx;

        // Project position onto the segment to get better accuracy
        while current_idx + 1 < self.waypoints.len() {
            let seg_start = self.waypoints[current_idx];
            let seg_end = self.waypoints[current_idx + 1];
            let seg_vec = seg_end - seg_start;
            let seg_len = seg_vec.length();

            if seg_len < 1e-6 {
                current_idx += 1;
                continue;
            }

            let to_pos = position - seg_start;
            let projection = to_pos.dot(seg_vec) / (seg_len * seg_len);
            let projection = projection.clamp(0.0, 1.0);

            let closest_on_seg = seg_start + seg_vec * projection;
            let dist_to_end = (seg_end - closest_on_seg).length();

            if accumulated_dist + dist_to_end >= look_ahead {
                // Target is on this segment
                let remaining = look_ahead - accumulated_dist;
                let t = remaining / dist_to_end;
                let target = closest_on_seg + (seg_end - closest_on_seg) * t;
                let heading = seg_vec.y.atan2(seg_vec.x);
                return Some((target, heading));
            }

            accumulated_dist += dist_to_end;
            current_idx += 1;
        }

        // If we've exhausted the path, return the last waypoint
        let last = *self.waypoints.last()?;
        let second_last = self.waypoints.get(self.waypoints.len() - 2)?;
        let heading = (last - *second_last).y.atan2((last - *second_last).x);
        Some((last, heading))
    }

    /// Interpolate waypoints at regular intervals
    pub fn interpolate(&self, num_points: usize) -> Vec<Vec2> {
        if self.waypoints.len() < 2 {
            return self.waypoints.clone();
        }

        // Compute total path length
        let mut total_length = 0.0;
        for i in 0..self.waypoints.len() - 1 {
            total_length += self.waypoints[i].distance(self.waypoints[i + 1]);
        }

        if total_length < 1e-6 {
            return self.waypoints.clone();
        }

        let step = total_length / (num_points - 1) as f32;
        let mut result = Vec::new();
        result.push(self.waypoints[0]);

        let mut current_length = 0.0;
        let mut target_length = step;
        let mut segment_idx = 0;

        while result.len() < num_points && segment_idx < self.waypoints.len() - 1 {
            let seg_start = self.waypoints[segment_idx];
            let seg_end = self.waypoints[segment_idx + 1];
            let seg_len = seg_start.distance(seg_end);

            if current_length + seg_len >= target_length {
                // Interpolate on this segment
                let t = (target_length - current_length) / seg_len;
                let point = seg_start.lerp(seg_end, t);
                result.push(point);
                target_length += step;
            } else {
                current_length += seg_len;
                segment_idx += 1;
            }
        }

        // Ensure last point is included
        if result.len() < num_points {
            result.push(*self.waypoints.last().unwrap());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trajectory_creation() {
        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        let traj = Trajectory::new(waypoints.clone());

        assert_eq!(traj.len(), 3);
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_find_closest_point() {
        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        let traj = Trajectory::new(waypoints);

        let (idx, dist) = traj.find_closest_point(Vec2::new(5.0, 1.0)).unwrap();
        assert_eq!(idx, 1);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_get_reference_point() {
        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        let traj = Trajectory::new(waypoints);

        let position = Vec2::new(2.0, 0.0);
        let (ref_point, heading) = traj.get_reference_point(position, 5.0).unwrap();

        // Reference point should be ahead on the path
        assert!(ref_point.x >= position.x);
        assert_relative_eq!(heading, 0.0, epsilon = 0.1); // Heading along x-axis
    }

    #[test]
    fn test_interpolate() {
        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let traj = Trajectory::new(waypoints);

        let interpolated = traj.interpolate(5);
        assert_eq!(interpolated.len(), 5);

        // First and last should match original waypoints
        assert_relative_eq!(interpolated[0].x, 0.0);
        assert_relative_eq!(interpolated[4].x, 10.0);

        // Middle points should be evenly spaced
        assert_relative_eq!(interpolated[2].x, 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_empty_trajectory() {
        let traj = Trajectory::new(vec![]);
        assert!(traj.is_empty());
        assert!(traj.find_closest_point(Vec2::ZERO).is_none());
    }

    #[test]
    fn test_single_waypoint() {
        let traj = Trajectory::new(vec![Vec2::new(5.0, 5.0)]);
        let (ref_point, _) = traj.get_reference_point(Vec2::ZERO, 1.0).unwrap();
        assert_relative_eq!(ref_point.x, 5.0);
        assert_relative_eq!(ref_point.y, 5.0);
    }
}

