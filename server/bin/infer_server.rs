use server::infer_server;
use rsproto::infer;
use slog::Drain;
use slog::Logger;
use slog::{info, o};

static RETINA_MODEL_PATH: &str = "./engine/retina_r50_v0.engine";
static ARCFACE_MODEL_PATH: &str = "./engine/arcface-r50_v0.engine";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_logger = build_logger(slog::Level::Info);

    let handler = infer_server::InferServer::new(
        RETINA_MODEL_PATH,
        ARCFACE_MODEL_PATH,
        root_logger.new(o!()),
    )
    .unwrap();
    let svc = infer::v1::infer_v1_server::InferV1Server::new(handler);
    let addr = "0.0.0.0:9989".parse().expect("invalid");
    info!(root_logger, "infer server is starting");
    tonic::transport::Server::builder()
        .add_service(svc)
        .serve(addr)
        .await?;
    info!(root_logger, "infer server has stopped");
    Ok(())
}

fn build_logger(log_level: slog::Level) -> Logger {
    // config logger
    let common_o = o!(
        "version" => env!("CARGO_PKG_VERSION"),
        "service" => "InferSErver",
    );

    let decorator = slog_term::TermDecorator::new().build();
    let base_drain = slog_term::CompactFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(base_drain)
        .build()
        .filter_level(log_level)
        .fuse();
    slog::Logger::root(drain, common_o)
}
