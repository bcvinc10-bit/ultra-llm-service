use axum::{
    routing::post,
    Json, Router,
};
use rayon::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

const EMBEDDING_DIM: usize = 64; 
const NUM_HEADS: usize = 4;      
const HEAD_DIM: usize = EMBEDDING_DIM / NUM_HEADS;
const NUM_NEURONS_FFN: usize = 156; 
const VOCAB_SIZE: usize = 256;    

// [Include Matrix, MultiHeadAttention, and UltraTransformerLLM structs & impls here...]

#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct GenerateResponse {
    response: String,
}

#[tokio::main]
async fn main() {
    // Initialize model once in an Arc for thread-safe shared state across Axum workers
    let model = std::sync::Arc::new(UltraTransformerLLM::new());
    println!("Model initialized successfully for web serving.");

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .with_state(model);

    // Render dynamically assigns a port via the PORT environment variable
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap();
    
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn generate_handler(
    axum::extract::State(model): axum::extract::State<std::sync::Arc<UltraTransformerLLM>>,
    Json(payload): Json<PromptRequest>,
) -> Json<GenerateResponse> {
    let max_tokens = payload.max_tokens.unwrap_or(32);
    let response = model.generate(&payload.prompt, max_tokens).unwrap_or_else(|_| "Generation error".into());
    
    Json(GenerateResponse { response })
}
