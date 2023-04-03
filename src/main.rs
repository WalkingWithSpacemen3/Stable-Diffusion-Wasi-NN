mod constants;

use std::env;
use std::fs;
use wasi_nn;
use rand::prelude::*;
use rand_distr::{ Distribution, StandardNormal };

fn main() {
    run_inference();
}

#[no_mangle]
fn run_inference() {
    let args: Vec<String> = env::args().collect();
    let model_file_name = &args[1];
    let prompt = &args[2];

    println!("Prompt: {}", prompt);

    let model_weights = fs::read(model_file_name).expect("Failed to read model file");
    println!("Read model weights, size in bytes: {}", model_weights.len());

    let graph = unsafe {
        wasi_nn
            ::load(
                &[&model_weights],
                wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE,
                wasi_nn::EXECUTION_TARGET_CPU
            )
            .expect("Failed to load graph")
    };
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    let context = unsafe {
        wasi_nn::init_execution_context(graph).expect("Failed to create wasi-nn execution context")
    };
    println!("Created wasi-nn execution context with ID: {}", context);

    let mut token = vec![49406, 33740, 8853, 539, 550, 18376, 6765, 320, 4558];
    token.extend(vec![49407; 77 - token.len()]);

    let token_u8 = i32_to_u8(&token);

    println!("Prompt length: {}", token_u8.len());

    let token_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &token_u8,
    };
    unsafe {
        wasi_nn::set_input(context, 0, token_tensor).unwrap();
    }

    let pos_ids: Vec<i32> = (0..77).collect();
    let pos_ids_u8 = i32_to_u8(&pos_ids);
    let pos_ids_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &pos_ids_u8,
    };
    unsafe {
        wasi_nn::set_input(context, 1, pos_ids_tensor).unwrap();
    }

    // Execute the inference.
    unsafe {
        wasi_nn::compute(context).expect("Failed to execute inference");
    }
    println!("Executed context inference");

    // Retrieve the output.
    let mut sd_context = vec![0f32; 59136];
    unsafe {
        wasi_nn
            ::get_output(
                context,
                0,
                &mut sd_context[..] as *mut [f32] as *mut u8,
                (sd_context.len() * 4).try_into().unwrap()
            )
            .expect("Failed to retrieve output");
    }
    let unconditional_token = constants::UNCONDITIONAL_TOKEN;
    let unconditional_token_u8 = i32_to_u8(&unconditional_token);
    let unconditional_token_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &unconditional_token_u8,
    };
    unsafe {
        wasi_nn::set_input(context, 0, unconditional_token_tensor).unwrap();
    }
    unsafe {
        wasi_nn::set_input(context, 1, pos_ids_tensor).unwrap();
    }
    unsafe {
        wasi_nn::compute(context).expect("Failed to execute inference");
    }
    println!("Executed unconditional context inference");
    let mut sd_unconditional_context = vec![0f32; 59136];
    unsafe {
        wasi_nn
            ::get_output(
                context,
                0,
                &mut sd_unconditional_context[..] as *mut [f32] as *mut u8,
                (sd_unconditional_context.len() * 4).try_into().unwrap()
            )
            .expect("Failed to retrieve output");
    }
    for output in sd_unconditional_context {
        println!("{}", output);
    }
    let num_steps = 50;
    let timesteps: Vec<usize> = (0..num_steps).map(|i| (i * 1000) / num_steps + 1).collect();
    let batch_size = 1;
    let img_height = 512;
    let img_width = 512;
    let n_h = img_height / 8;
    let n_w = img_width / 8;
    let alphas: Vec<f32> = timesteps
        .iter()
        .map(|&t| constants::ALPHAS_CUMPROD[t])
        .collect();
    let alphas_prev: Vec<f32> = vec![1.0]
        .iter()
        .chain(alphas.iter().take(timesteps.len() - 1))
        .cloned()
        .collect();
    let latent = generate_normal_numbers(batch_size * n_h * n_w * 4);
    let latent_u8 = f32_to_u8(&latent);
    let latent_tensor = wasi_nn::Tensor {
        dimensions: &[1, 64, 64, 4],
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &latent_u8,
    };
}

fn i32_to_u8(data: &[i32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &i in data {
        let bytes = i.to_ne_bytes();
        result.extend(&bytes);
    }
    result
}

fn f32_to_u8(data: &[f32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &i in data {
        let bytes = i.to_ne_bytes();
        result.extend(&bytes);
    }
    result
}

fn generate_normal_numbers(n: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    let normal = StandardNormal;
    let mut vec: Vec<f32> = Vec::new();
    for _ in 0..n {
        vec.push(normal.sample(&mut rng));
    }
    vec
}

fn timestep_embedding(timestep: &usize, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let freqs = (0..half).map(|i| { (-((i as f32) * (max_period.ln() / (half as f32)))).exp() });
    let cos_embedding = freqs
        .clone()
        .map(|f| f * (*timestep as f32))
        .map(|arg| arg.cos());
    let sin_embedding = freqs.map(|f| f * (*timestep as f32)).map(|arg| arg.sin());
    let embedding = cos_embedding.chain(sin_embedding).collect::<Vec<_>>();
    embedding
}