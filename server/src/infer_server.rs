use crate::face;
use bytes::Bytes;
use rsproto::v1;
use slog::info;
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct InferServer {
    retina_arc_face: face::RetinaArcFace,
    // discoverer: Discoverer,
    logger: slog::Logger,
}

unsafe impl Send for InferServer {}
unsafe impl Sync for InferServer {}

impl InferServer {
    pub fn new(
        retina_engine_path: &str,
        arc_engine_path: &str,
        logger: slog::Logger,
    ) -> Result<InferServer, Status> {
        let raf = face::RetinaArcFace::new(retina_engine_path, arc_engine_path)?;
        Ok(InferServer {
            retina_arc_face: raf,
            logger,
        })
    }
}

#[tonic::async_trait]
impl v1::infer::infer_v1_server::InferV1 for InferServer {
    async fn retina_arc_face_file(
        &self,
        request: Request<v1::infer::RetinaArcFaceFileRequest>,
    ) -> Result<Response<v1::infer::RetinaArcFaceFileResponse>, Status> {
        let img_path = request.into_inner().path;
        info!(self.logger, "retina_arc_face_file"; "img_path" => &img_path.as_str());

        let raf = self.retina_arc_face.clone();
        let det = tokio::task::spawn_blocking(move || raf.process_image_file(img_path))
            .await
            .map_err(resp_join_err)??;
        let resp = v1::infer::RetinaArcFaceFileResponse { face: Some(det) };
        Ok(Response::new(resp))
    }

    async fn retina_arc_face_blob(
        &self,
        request: Request<v1::infer::RetinaArcFaceBlobRequest>,
    ) -> Result<Response<v1::infer::RetinaArcFaceFileResponse>, Status> {
        let blob = Bytes::from(request.into_inner().blob.to_vec());
        let raf = self.retina_arc_face.clone();
        let det = tokio::task::spawn_blocking(move || raf.process_image_blob(blob))
            .await
            .map_err(resp_join_err)??;
        let resp = v1::infer::RetinaArcFaceFileResponse { face: Some(det) };
        Ok(Response::new(resp))
    }
}

fn resp_join_err(e: tokio::task::JoinError) -> Status {
    Status::unknown(format!("unexpected join error: {:?}", e))
}
