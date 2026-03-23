use kornia_algebra::SE3F32;
use crate::frontend::FrameFeatures;

/// Map representation and landmark management.
#[derive(Clone, Debug)]
pub struct Landmark {
    pub id: usize,
    pub position: [f32; 3],
}

/// A keyframe in the SLAM map.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub id: usize,
    pub pose: SE3F32,
    pub features: FrameFeatures,
    pub landmarks: Vec<usize>, // IDs of landmarks seen in this keyframe
}

pub struct Map {
    pub landmarks: Vec<Landmark>,
    pub keyframes: Vec<Keyframe>,
    next_landmark_id: usize,
    next_keyframe_id: usize,
}

impl Map {
    pub fn new() -> Self {
        Self {
            landmarks: Vec::new(),
            keyframes: Vec::new(),
            next_landmark_id: 0,
            next_keyframe_id: 0,
        }
    }

    pub fn add_landmark(&mut self, point: [f32; 3]) -> usize {
        let id = self.next_landmark_id;
        self.next_landmark_id += 1;
        self.landmarks.push(Landmark { id, position: point });
        id
    }

    pub fn add_keyframe(&mut self, pose: SE3F32, features: FrameFeatures, landmarks: Vec<usize>) -> usize {
        let id = self.next_keyframe_id;
        self.next_keyframe_id += 1;
        self.keyframes.push(Keyframe { id, pose, features, landmarks });
        id
    }

    pub fn nearby_landmarks(&self, pose: [f32; 3], radius: f32) -> Vec<&Landmark> {
        self.landmarks
            .iter()
            .filter(|lm| {
                let dx = lm.position[0] - pose[0];
                let dy = lm.position[1] - pose[1];
                let dz = lm.position[2] - pose[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            })
            .collect()
    }
}
