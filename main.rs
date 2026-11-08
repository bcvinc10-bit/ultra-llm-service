use axum::{
    routing::post,
    Json, Router,
};
use rayon::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::error::Error;
use std::fmt;

// Hyperparameters scaled for a localized 10,000-parameter ultra architecture
const EMBEDDING_DIM: usize = 64; 
const NUM_HEADS: usize = 4;      
const HEAD_DIM: usize = EMBEDDING_DIM / NUM_HEADS;
const NUM_NEURONS_FFN: usize = 156; 
const VOCAB_SIZE: usize = 256;    

#[derive(Debug)]
enum ModelError {
    DimensionMismatch { expected: String, found: String },
    BufferOverflow { expected: usize, found: usize },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, found {}", expected, found)
            }
            ModelError::BufferOverflow { expected, found } => {
                write!(f, "Buffer size mismatch: expected {} bytes, found {} bytes", expected, found)
            }
        }
    }
}

impl Error for ModelError {}

/// High-performance parallel Matrix structure optimized via Rayon
#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (rows + cols) as f32).sqrt();
        let data = (0..rows * cols)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self { rows, cols, data }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    fn matmul(&self, other: &Matrix) -> Result<Matrix, ModelError> {
        if self.cols != other.rows {
            return Err(ModelError::DimensionMismatch {
                expected: format!("self.cols ({}) == other.rows ({})", self.cols, other.rows),
                found: "mismatch".to_string(),
            });
        }
        let mut result = Matrix::zeros(self.rows, other.cols);

        result.data.par_chunks_mut(other.cols).enumerate().for_each(|(i, row_slice)| {
            for k in 0..self.cols {
                let self_val = self.data[i * self.cols + k];
                if self_val == 0.0 { continue; }
                let other_row_start = k * other.cols;
                for j in 0..other.cols {
                    row_slice[j] += self_val * other.data[other_row_start + j];
                }
            }
        });
        Ok(result)
    }
}

/// Multi-Head Self-Attention Layer
struct MultiHeadAttention {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
}

impl MultiHeadAttention {
    fn new() -> Self {
        Self {
            w_q: Matrix::new(EMBEDDING_DIM, EMBEDDING_DIM),
            w_k: Matrix::new(EMBEDDING_DIM, EMBEDDING_DIM),
            w_v: Matrix::new(EMBEDDING_DIM, EMBEDDING_DIM),
            w_o: Matrix::new(EMBEDDING_DIM, EMBEDDING_DIM),
        }
    }

    fn forward(&self, x: &Matrix) -> Result<Matrix, ModelError> {
        let seq_len = x.rows;
        let q = x.matmul(&self.w_q)?;
        let k = x.matmul(&self.w_k)?;
        let v = x.matmul(&self.v)?;

        let mut attention_out = Matrix::zeros(seq_len, EMBEDDING_DIM);

        attention_out.data.par_chunks_mut(seq_len * HEAD_DIM).enumerate().for_each(|(h, head_slice)| {
            let scale = (HEAD_DIM as f32).sqrt();
            
            for i in 0..seq_len {
                let mut scores = vec![0.0; seq_len];
                let mut max_score = f32::NEG_INFINITY;

                for j in 0..seq_len {
                    let mut dot = 0.0;
                    for d in 0..HEAD_DIM {
                        let q_val = q.data[i * EMBEDDING_DIM + h * HEAD_DIM + d];
                        let k_val = k.data[j * EMBEDDING_DIM + h * HEAD_DIM + d];
                        dot += q_val * k_val;
                    }
                    let score = dot / scale;
                    scores[j] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut sum_exp = 0.0;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                for score in &mut scores {
                    *score /= sum_exp;
                }

                for d in 0..HEAD_DIM {
                    let mut weighted_sum = 0.0;
                    for j in 0..seq_len {
                        let v_val = v.data[j * EMBEDDING_DIM + h * HEAD_DIM + d];
                        weighted_sum += scores[j] * v_val;
                    }
                    head_slice[i * HEAD_DIM + d] = weighted_sum;
                }
            }
        });

        attention_out.matmul(&self.w_o)
    }
}

/// Advanced Chat Transformer Model Container
struct UltraTransformerLLM {
    embedding: Matrix,
    mha: MultiHeadAttention,
    ffn_w1: Matrix,
    ffn_w2: Matrix,
    output_projection: Matrix,
}

impl UltraTransformerLLM {
    fn new() -> Self {
        Self {
            embedding: Matrix::new(VOCAB_SIZE, EMBEDDING_DIM),
            mha: MultiHeadAttention::new(),
            ffn_w1: Matrix::new(EMBEDDING_DIM, NUM_NEURONS_FFN),
            ffn_w2: Matrix::new(NUM_NEURONS_FFN, EMBEDDING_DIM),
            output_projection: Matrix::new(EMBEDDING_DIM, VOCAB_SIZE),
        }
    }

    fn embed_tokens(&self, tokens: &[u8]) -> Matrix {
        let mut x = Matrix::zeros(tokens.len(), EMBEDDING_DIM);
        x.data.par_chunks_mut(EMBEDDING_DIM).enumerate().for_each(|(i, row)| {
            let token_id = tokens[i] as usize;
            for d in 0..EMBEDDING_DIM {
                row[d] = self.embedding.data[token_id * EMBEDDING_DIM + d];
            }
        });
        x
    }

    fn forward(&self, tokens: &[u8]) -> Result<Matrix, ModelError> {
        let x0 = self.embed_tokens(tokens);

        let mha_out = self.mha.forward(&x0)?;
        let mut x1 = Matrix::zeros(x0.rows, x0.cols);
        x1.data.par_iter_mut().enumerate().for_each(|(idx, val)| {
            *val = x0.data[idx] + mha_out.data[idx];
        });

        let hidden = x1.matmul(&self.ffn_w1)?;
        let mut activated = Matrix::zeros(hidden.rows, hidden.cols);
        activated.data.par_iter_mut().enumerate().for_each(|(idx, val)| {
            let h = hidden.data[idx];
            *val = 0.5 * h * (1.0 + (h * 0.797885 * (1.0 + 0.044715 * h * h)).tanh());
        });

        let ffn_out = activated.matmul(&self.ffn_w2)?;
        
        let mut x2 = Matrix::zeros(x1.rows, x1.cols);
        x2.data.par_iter_mut().enumerate().for_each(|(idx, val)| {
            *val = x1.data[idx] + ffn_out.data[idx];
        });

        x2.matmul(&self.output_projection)
    }

    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, ModelError> {
        let mut tokens: Vec<u8> = prompt.bytes().collect();
        let mut response_bytes = Vec::new();

        for _ in 0..max_tokens {
            let logits = self.forward(&tokens)?;
            let last_row_start = (tokens.len() - 1) * VOCAB_SIZE;
            let last_logits = &logits.data[last_row_start..last_row_start + VOCAB_SIZE];

            let next_token = sample_token(last_logits, 0.8);
            
            if next_token == 0 || next_token == b'\n' {
                break;
            }

            response_bytes.push(next_token);
            tokens.push(next_token);
        }

        Ok(String::from_utf8_lossy(&response_bytes).into_owned())
    }
}

fn sample_token(logits: &[f32], temperature: f32) -> u8 {
    let mut rng = rand::thread_rng();
    let scaled_logits: Vec<f32> = logits.iter().map(|&l| (l / temperature).exp()).collect();
    let sum: f32 = scaled_logits.iter().sum();
    
    if sum == 0.0 || sum.is_nan() {
        return b' ';
    }

    let mut choice = rng.gen<f32>() * sum;
    for (i, &prob) in scaled_logits.iter().enumerate() {
        if choice < prob {
            return i as u8;
        }
        choice -= prob;
    }
    b' '
}

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
    let model = std::sync::Arc::new(UltraTransformerLLM::new());
    println!("Model initialized successfully for web serving.");

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .with_state(model);

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
