mod compute;
use axum::{
    routing::{get, post},
    Router,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("🚀 Starting the server");
    log::info!("🔥 Listening on: http://0.0.0.0:3000");
    log::info!("🔧 Press Ctrl+C to stop the server");

    // build our application with a single route
    let app = Router::new()
        .route("/", get(|| async { "Welcome to Kornia!" }))
        .route("/api/v0/compute/:mean_std", post(compute::compute_mean_std));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await?;

    Ok(())
}
