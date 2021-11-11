use crate::errors::ToStatus;
use crate::face_preprocess;
use bytes::Bytes;
use opencv::{core, imgcodecs, imgproc, prelude::*, Result};
use retinafacers::{arcface, retinaface};
use rsproto::v1;
use std::sync::{Arc, Mutex};
use tonic::Status;

#[derive(Debug, Clone)]
pub struct RetinaArcFace {
    retina_engine: Arc<Mutex<retinaface::Retinaface>>,
    arc_engine: Arc<Mutex<arcface::Arcface>>,
}

unsafe impl Send for RetinaArcFace {}
unsafe impl Sync for RetinaArcFace {}

impl RetinaArcFace {
    pub fn new(retina_engine_path: &str, arc_engine_path: &str) -> Result<RetinaArcFace, Status> {
        let r_engine = retinaface::Retinaface::new(retina_engine_path, false);
        let a_engine = arcface::Arcface::new(arc_engine_path, false);
        Ok(RetinaArcFace {
            retina_engine: Arc::new(Mutex::new(r_engine)),
            arc_engine: Arc::new(Mutex::new(a_engine)),
        })
    }

    pub fn process_image_file(&self, img_path: String) -> Result<v1::face::RetinaArcFace, Status> {
        let img = imgcodecs::imread(img_path.as_str(), imgcodecs::IMREAD_COLOR)
            .map_err(|e| e.to_status())?;
        self.process_image(&img)
    }

    pub fn process_image_blob(&self, data: Bytes) -> Result<v1::face::RetinaArcFace, Status> {
        let src = Mat::from_slice::<u8>(data.as_ref()).map_err(|e| e.to_status())?;
        let img =
            imgcodecs::imdecode(&src, imgcodecs::IMREAD_UNCHANGED).map_err(|e| e.to_status())?;
        self.process_image(&img)
    }

    // process_image detect faces in the images and also provides arc face features
    pub fn process_image(&self, img: &Mat) -> Result<v1::face::RetinaArcFace, Status> {
        let img_size = img.size().map_err(|e| e.to_status())?;
        let pr_img = preprocess_img(img).map_err(|e| e.to_status())?;

        let values = prepare_retinaface_input_data(&pr_img).map_err(|e| e.to_status())?;

        let mut retina_detection =
            self.retina_engine
                .lock()
                .unwrap()
                .infer(values, img_size.height, img_size.width, 0.01);

        let mut res = v1::face::RetinaArcFace {
            original_image_height: img_size.height,
            original_image_width: img_size.width,
            faces: vec![],
        };

        if retina_detection.size == 0 {
            return Ok(res);
        }

        for i in 0..retina_detection.size {
            let det = retina_detection.at(i);

            // gt face landmark
            let v1 = [
                [30.2946_f32, 51.6963_f32],
                [65.5318_f32, 51.5014_f32],
                [48.0252_f32, 71.7366_f32],
                [33.5493_f32, 92.3655_f32],
                [62.7299_f32, 92.2041_f32],
            ];

            let src_mat = core::Mat::from_slice_2d(&v1).map_err(|e| e.to_status())?;

            // align for face lan
            let v2 = [
                [det.landmark[0], det.landmark[1]],
                [det.landmark[2], det.landmark[3]],
                [det.landmark[4], det.landmark[5]],
                [det.landmark[6], det.landmark[7]],
                [det.landmark[8], det.landmark[9]],
            ];

            let dst_mat = core::Mat::from_slice_2d(&v2).map_err(|e| e.to_status())?;

            let m = face_preprocess::similar_transform(&dst_mat, &src_mat)
                .map_err(|e| e.to_status())?;

            let mut aligned = img.clone();

            imgproc::warp_perspective(
                &img,
                &mut aligned,
                &m,
                core::Size::new(96, 112),
                imgproc::INTER_LINEAR,
                core::BORDER_CONSTANT,
                core::Scalar::all(0.0),
            )
            .map_err(|e| e.to_status())?;

            imgproc::resize(
                &aligned.clone(),
                &mut aligned,
                core::Size::new(112, 112),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .map_err(|e| e.to_status())?;

            // prepare arcface infer input data

            let arc_data = prepare_arcface_input_data(&aligned).map_err(|e| e.to_status())?;
            let arcface_res = self.arc_engine.lock().unwrap().infer(arc_data);
            let arc_slice = arcface_res.x.to_vec();

            let arc_mat = Mat::from_slice(&arc_slice).map_err(|e| e.to_status())?;

            let mut normal_mat = Mat::default().map_err(|e| e.to_status())?;
            core::normalize(
                &arc_mat,
                &mut normal_mat,
                1.0,
                0.0,
                core::NORM_L2,
                -1,
                &core::no_array().map_err(|e| e.to_status())?,
            )
            .map_err(|e| e.to_status())?;
            let normal_slice = normal_mat.at_row::<f32>(0).map_err(|e| e.to_status())?;

            let face_detection = v1::face::RetinafaceDetection {
                bbox: Some(v1::face::BBox {
                    x1: det.box_x1(),
                    y1: det.box_y1(),
                    x2: det.box_x2(),
                    y2: det.box_y2(),
                }),
                class_confidence: det.class_confidence,
                landmarks: Some(v1::face::RetinafaceLandmark {
                    left_eye_x: det.landmark_left_eye().0,
                    left_eye_y: det.landmark_left_eye().1,
                    right_eye_x: det.landmark_right_eye().0,
                    right_eye_y: det.landmark_right_eye().1,
                    nose_x: det.landmark_nose().0,
                    nose_y: det.landmark_nose().1,
                    mouth_left_corner_x: det.landmark_mouth_left().0,
                    mouth_left_corner_y: det.landmark_mouth_left().1,
                    mouth_right_corner_x: det.landmark_mouth_right().0,
                    mouth_right_corner_y: det.landmark_mouth_right().1,
                }),
                arcface_feature: normal_slice.to_vec(),
                retina_engine_version: 1,
                arcface_engine_version: 1,
            };

            res.faces.push(face_detection);
        }

        Ok(res)
    }
}

fn preprocess_img(img: &Mat) -> Result<Mat, opencv::Error> {
    let w;
    let h;
    let x;
    let y;
    let r_w = (retinaface::INPUT_W as f32) / (img.cols() as f32);
    let r_h = retinaface::INPUT_H as f32 / (img.rows() as f32);
    if r_h > r_w {
        w = retinaface::INPUT_W;
        h = (r_w * (img.rows() as f32)) as i32;
        x = 0;
        y = (retinaface::INPUT_H - h) / 2;
    } else {
        w = (r_h * (img.cols() as f32)) as i32;
        h = retinaface::INPUT_H;
        x = (retinaface::INPUT_W - w) / 2;
        y = 0;
    }
    let mut re = unsafe { Mat::new_rows_cols(h, w, core::CV_8UC3)? };
    let re_size = re.size()?;

    imgproc::resize(img, &mut re, re_size, 0.0, 0.0, imgproc::INTER_CUBIC)?;
    let out = Mat::new_rows_cols_with_default(
        retinaface::INPUT_H,
        retinaface::INPUT_W,
        core::CV_8UC3,
        core::Scalar::all(128.0),
    )?;
    let mut out_roi = Mat::roi(&out, core::Rect::new(x, y, re.cols(), re.rows()))?;
    re.copy_to(&mut out_roi)?;

    Ok(out)
}

fn prepare_retinaface_input_data(
    img: &Mat,
) -> Result<Vec<f32>, opencv::Error> {
    let mut values = vec![
        0_f32;
        (retinaface::BATCH_SIZE * 3 * retinaface::INPUT_H * retinaface::INPUT_W)
            as usize
    ];

    for b in 0..retinaface::BATCH_SIZE {
        let bidx =
            (b as usize) * 3 * (retinaface::INPUT_H as usize) * (retinaface::INPUT_W as usize);
        for i in 0..(retinaface::INPUT_H * retinaface::INPUT_W) as usize {
            values[bidx + i] = img.at::<core::Vec3b>(i as i32)?[0] as f32 - 104.0;
            values[bidx + i + (retinaface::INPUT_H * retinaface::INPUT_W) as usize] =
                img.at::<core::Vec3b>(i as i32)?[1] as f32 - 117.0;
            values[bidx + i + 2 * (retinaface::INPUT_H * retinaface::INPUT_W) as usize] =
                img.at::<core::Vec3b>(i as i32)?[2] as f32 - 123.0;
        }
    }
    Ok(values)
}

fn prepare_arcface_input_data(img: &Mat) -> Result<Vec<f32>, opencv::Error> {
    let mut values =
        vec![0_f32; (arcface::BATCH_SIZE * 3 * arcface::INPUT_H * arcface::INPUT_W) as usize];
    for b in 0..(arcface::BATCH_SIZE) as usize {
        for i in 0..(arcface::INPUT_H * arcface::INPUT_W) as usize {
            values[b + i] = (img.at::<core::Vec3b>(i as i32)?[2] as f32 - 127.5) * 0.0078125;
            values[b + i + (arcface::INPUT_H * arcface::INPUT_W) as usize] =
                (img.at::<core::Vec3b>(i as i32)?[1] as f32 - 127.5) * 0.0078125;
            values[b + i + 2 * (arcface::INPUT_H * arcface::INPUT_W) as usize] =
                (img.at::<core::Vec3b>(i as i32)?[0] as f32 - 127.5) * 0.0078125;
        }
    }

    Ok(values)
}
