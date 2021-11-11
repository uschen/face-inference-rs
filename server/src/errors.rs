use tonic::Status;

pub trait ToStatus {
    fn to_status(&self) -> Status;
}

impl ToStatus for opencv::Error {
    fn to_status(&self) -> Status {
        Status::internal(format!("opencv error: {:?}", self))
    }
}
