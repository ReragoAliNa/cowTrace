use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct BBox {
    #[pyo3(get, set)]
    pub x1: f32,
    #[pyo3(get, set)]
    pub y1: f32,
    #[pyo3(get, set)]
    pub x2: f32,
    #[pyo3(get, set)]
    pub y2: f32,
}

#[pymethods]
impl BBox {
    #[new]
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        BBox { x1, y1, x2, y2 }
    }
}

// Compute Intersection over Union (IoU)
fn box_iou(box_a: &BBox, box_b: &BBox) -> f32 {
    let xi1 = box_a.x1.max(box_b.x1);
    let yi1 = box_a.y1.max(box_b.y1);
    let xi2 = box_a.x2.min(box_b.x2);
    let yi2 = box_a.y2.min(box_b.y2);

    let inter_w = (xi2 - xi1).max(0.0);
    let inter_h = (yi2 - yi1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1);
    let area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

#[pyclass]
pub struct RustTracker {
    max_age: usize,
    min_iou: f32,
    next_id: i32,
    track_ids: Vec<i32>,
    active_boxes: Vec<BBox>,
    time_since_update: Vec<usize>,
}

#[pymethods]
impl RustTracker {
    #[new]
    fn new(max_age: usize, min_iou: f32) -> Self {
        RustTracker {
            max_age,
            min_iou,
            next_id: 0,
            track_ids: Vec::new(),
            active_boxes: Vec::new(),
            time_since_update: Vec::new(),
        }
    }

    fn update(&mut self, detections: Vec<BBox>) -> PyResult<Vec<i32>> {
        let num_tracks = self.active_boxes.len();
        let num_dets = detections.len();
        let mut assigned_ids = vec![-1; num_dets];

        // Increment age of all existing tracks
        for age in self.time_since_update.iter_mut() {
            *age += 1;
        }

        if num_tracks == 0 {
            // First frame: initialize all detections as new trackers
            for i in 0..num_dets {
                let id = self.next_id;
                self.next_id += 1;
                self.track_ids.push(id);
                self.active_boxes.push(detections[i]);
                self.time_since_update.push(0);
                assigned_ids[i] = id;
            }
            return Ok(assigned_ids);
        }

        // Calculate Cost Matrix (1.0 - IoU)
        let mut cost_matrix = vec![vec![1.0; num_dets]; num_tracks];
        cost_matrix.par_iter_mut().enumerate().for_each(|(t_idx, row)| {
            for d_idx in 0..num_dets {
                let iou = box_iou(&self.active_boxes[t_idx], &detections[d_idx]);
                row[d_idx] = 1.0 - iou;
            }
        });

        // Greedy matching
        let mut matched_dets = vec![false; num_dets];
        let mut matched_tracks = vec![false; num_tracks];

        // We search for matches with highest IoU (minimum cost)
        let threshold = 1.0 - self.min_iou;
        loop {
            let mut min_val = 1.0;
            let mut best_t = None;
            let mut best_d = None;

            for t in 0..num_tracks {
                if matched_tracks[t] {
                    continue;
                }
                for d in 0..num_dets {
                    if matched_dets[d] {
                        continue;
                    }
                    if cost_matrix[t][d] < min_val {
                        min_val = cost_matrix[t][d];
                        best_t = Some(t);
                        best_d = Some(d);
                    }
                }
            }

            if let (Some(t), Some(d)) = (best_t, best_d) {
                if min_val < threshold {
                    matched_tracks[t] = true;
                    matched_dets[d] = true;
                    self.active_boxes[t] = detections[d];
                    self.time_since_update[t] = 0;
                    assigned_ids[d] = self.track_ids[t];
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Spawn new trackers for unmatched detections
        for d in 0..num_dets {
            if !matched_dets[d] {
                let id = self.next_id;
                self.next_id += 1;
                self.track_ids.push(id);
                self.active_boxes.push(detections[d]);
                self.time_since_update.push(0);
                assigned_ids[d] = id;
            }
        }

        // Clean up dead trackers (exceeding max_age)
        let mut keep_ids = Vec::new();
        let mut keep_boxes = Vec::new();
        let mut keep_ages = Vec::new();

        for i in 0..self.active_boxes.len() {
            if self.time_since_update[i] < self.max_age {
                keep_ids.push(self.track_ids[i]);
                keep_boxes.push(self.active_boxes[i]);
                keep_ages.push(self.time_since_update[i]);
            }
        }

        self.track_ids = keep_ids;
        self.active_boxes = keep_boxes;
        self.time_since_update = keep_ages;

        Ok(assigned_ids)
    }
}

#[pymodule]
fn cow_trace_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BBox>()?;
    m.add_class::<RustTracker>()?;
    Ok(())
}
