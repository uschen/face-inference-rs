#![allow(dead_code)]

pub mod v1 {
    pub mod infer {
        include!("prost/v1.infer.rs");
    }
    pub mod face {
        include!("prost/v1.face.rs");
    }
}
