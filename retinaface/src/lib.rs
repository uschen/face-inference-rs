#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod bindings;

pub mod retinaface {
    use crate::bindings;
    use std::ffi::CString;

    pub const INPUT_H: i32 = 480;
    pub const INPUT_W: i32 = 640;

    pub const BATCH_SIZE: i32 = 1;

    pub type Detection = bindings::retinaface_Detection;

    impl Detection {
        pub fn box_x1(&self) -> i32 {
            self.bbox[0] as i32
        }
        pub fn box_y1(&self) -> i32 {
            self.bbox[1] as i32
        }
        pub fn box_x2(&self) -> i32 {
            self.bbox[2] as i32
        }
        pub fn box_y2(&self) -> i32 {
            self.bbox[3] as i32
        }
        pub fn box_width(&self) -> i32 {
            (self.bbox[2] - self.bbox[0]) as i32
        }
        pub fn box_height(&self) -> i32 {
            (self.bbox[3] - self.bbox[1]) as i32
        }
        pub fn landmark_left_eye(&self) -> (f32, f32) {
            (self.landmark[0], self.landmark[1])
        }
        pub fn landmark_right_eye(&self) -> (f32, f32) {
            (self.landmark[2], self.landmark[3])
        }
        pub fn landmark_nose(&self) -> (f32, f32) {
            (self.landmark[4], self.landmark[5])
        }
        pub fn landmark_mouth_left(&self) -> (f32, f32) {
            (self.landmark[6], self.landmark[7])
        }
        pub fn landmark_mouth_right(&self) -> (f32, f32) {
            (self.landmark[8], self.landmark[9])
        }
    }

    pub type Detections = bindings::retinaface_Detections;

    pub type InputData = [f32; BATCH_SIZE as usize * 3 * INPUT_H as usize * INPUT_W as usize];

    impl Detections {
        pub fn at(&mut self, idx: i32) -> Detection {
            unsafe { bindings::retinaface_Detections_at(self, idx) }
        }
        pub fn size(&mut self) -> i32 {
            unsafe { bindings::retinaface_Detections_size(self) }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Retinaface {
        engine: *mut bindings::Retinaface,
    }

    impl Retinaface {
        pub fn new(path: &str, verbose: bool) -> Retinaface {
            let c_str = CString::new(path).unwrap();
            let c_world: *mut std::os::raw::c_char = c_str.as_ptr() as *mut std::os::raw::c_char;
            let engine: *mut bindings::Retinaface;
            unsafe {
                engine = bindings::Retinaface_new(c_world, verbose);
            };
            // Box::new(Retinaface { engine })
            Retinaface { engine }
        }

        // # Safety
        pub fn infer(
            &mut self,
            input: Vec<f32>,
            org_img_h: i32,
            org_img_w: i32,
            vis_thresh: f32,
        ) -> Box<Detections> {
            let mut det = Box::new(Detections {
                size: 0,
                x: [0.0; 189001usize],
            });
            unsafe {
                bindings::Retinaface_infer(
                    self.engine,
                    det.as_mut(),
                    input.into_boxed_slice().as_mut_ptr(),
                    org_img_h,
                    org_img_w,
                    vis_thresh,
                );
            }
            det
        }
    }
}

pub mod arcface {
    use crate::bindings;
    use std::ffi::CString;

    pub const INPUT_H: i32 = 112;
    pub const INPUT_W: i32 = 112;

    pub const BATCH_SIZE: i32 = 1;

    pub type InputData = [f32; BATCH_SIZE as usize * 3 * INPUT_H as usize * INPUT_W as usize];

    pub type Detection = bindings::arcface_Detection;

    #[derive(Debug, Clone)]
    pub struct Arcface {
        engine: *mut bindings::Arcface,
    }

    impl Arcface {
        pub fn new(path: &str, verbose: bool) -> Arcface {
            let c_str = CString::new(path).unwrap();
            let c_world: *mut std::os::raw::c_char = c_str.as_ptr() as *mut std::os::raw::c_char;
            let engine: *mut bindings::Arcface;
            unsafe {
                engine = bindings::Arcface_new(c_world, verbose);
            };
            Arcface { engine }
        }

        pub fn infer(&mut self, input: Vec<f32>) -> Detection {
            let mut det = Detection { x: [0.0; 512usize] };
            unsafe {
                bindings::Arcface_infer(
                    self.engine,
                    &mut det,
                    input.into_boxed_slice().as_mut_ptr(),
                );
            }
            det
        }
    }

}
