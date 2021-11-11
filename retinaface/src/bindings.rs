#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct retinaface_Detection {
    pub bbox: [f32; 4usize],
    pub class_confidence: f32,
    pub landmark: [f32; 10usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct retinaface_Detections {
    pub size: ::std::os::raw::c_int,
    pub x: [f32; 189001usize],
}

// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn retinaface_Detections_size(self_: *mut retinaface_Detections) -> ::std::os::raw::c_int;
}
// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn retinaface_Detections_at(
        self_: *mut retinaface_Detections,
        idx: ::std::os::raw::c_int,
    ) -> retinaface_Detection;
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Retinaface {
    _unused: [u8; 0],
}

// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn Retinaface_new(path: *mut ::std::os::raw::c_char, verbose: bool) -> *mut Retinaface;
}
// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn Retinaface_destroy(self_: *mut Retinaface);
}

// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn Retinaface_infer(
        self_: *mut Retinaface,
        output: *mut retinaface_Detections,
        input: *const f32,
        org_img_h: i32,
        org_img_w: i32,
        vis_thresh: f32,
    );
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct arcface_Detection {
    pub x: [f32; 512usize],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Arcface {
    _unused: [u8; 0],
}

// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn Arcface_new(path: *mut ::std::os::raw::c_char, verbose: bool) -> *mut Arcface;
}
// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn Arcface_destroy(self_: *mut Arcface);
}

// #[link(name = "retinaface", kind = "static")]
extern "C" {
    pub fn Arcface_infer(self_: *mut Arcface, output: *mut arcface_Detection, input: *const f32);
}
