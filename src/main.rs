mod constants;

use std::env;
use std::fs;
use std::path::PathBuf;
use wasi_nn;
use rand::prelude::*;
use rand_distr::{ Distribution, StandardNormal };

fn main() {
    run_inference();
}

#[no_mangle]
fn run_inference() {
    let args: Vec<String> = env::args().collect();
    let model_folder = &args[1];
    let text_encoder_path = PathBuf::from(format!("{}/text_encoder.tflite", model_folder));
    let diffusion_model_path = PathBuf::from(format!("{}/diffusion_model.tflite", model_folder));
    let decoder_path = PathBuf::from(format!("{}/decoder.tflite", model_folder));
    let prompt = &args[2];

    println!("Prompt: {}", prompt);

    let text_encoder_weights = fs
        ::read(text_encoder_path)
        .expect("Failed to read model text encoder");
    println!("Read text encoder weights, size in bytes: {}", text_encoder_weights.len());
    let diffusion_model_weights = fs
        ::read(diffusion_model_path)
        .expect("Failed to read model diffusion model");
    println!("Read diffusion model weights, size in bytes: {}", diffusion_model_weights.len());
    let decoder_weights = fs::read(decoder_path).expect("Failed to read model decoder");
    println!("Read decoder weights, size in bytes: {}", decoder_weights.len());
    let text_encoder_graph = unsafe {
        wasi_nn
            ::load(
                &[&text_encoder_weights],
                wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE,
                wasi_nn::EXECUTION_TARGET_CPU
            )
            .expect("Failed to load graph")
    };
    println!("Loaded graph into wasi-nn with ID: {}", text_encoder_graph);
    let diffusion_model_graph = unsafe {
        wasi_nn
            ::load(
                &[&diffusion_model_weights],
                wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE,
                wasi_nn::EXECUTION_TARGET_CPU
            )
            .expect("Failed to load graph")
    };
    println!("Loaded graph into wasi-nn with ID: {}", diffusion_model_graph);
    let decoder_graph = unsafe {
        wasi_nn
            ::load(
                &[&decoder_weights],
                wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE,
                wasi_nn::EXECUTION_TARGET_CPU
            )
            .expect("Failed to load graph")
    };
    println!("Loaded graph into wasi-nn with ID: {}", decoder_graph);

    let text_encoder = unsafe {
        wasi_nn
            ::init_execution_context(text_encoder_graph)
            .expect("Failed to create wasi-nn execution context")
    };
    println!("Created text encoder execution context with ID: {}", text_encoder);

    let decoder = unsafe {
        wasi_nn
            ::init_execution_context(decoder_graph)
            .expect("Failed to create wasi-nn execution context")
    };
    println!("Created decoder execution context with ID: {}", decoder);

    let diffusion_model = unsafe {
        wasi_nn
            ::init_execution_context(diffusion_model_graph)
            .expect("Failed to create wasi-nn execution context")
    };
    println!("Created diffusion model execution context with ID: {}", diffusion_model);

    let mut token = vec![49406, 33740, 8853, 539, 550, 18376, 6765, 320, 4558];
    token.extend(vec![49407; 77 - token.len()]);
    let token_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &i32_to_u8(&token),
    };
    unsafe {
        wasi_nn::set_input(text_encoder, 0, token_tensor).unwrap();
    }

    let pos_ids: Vec<i32> = (0..77).collect();
    let pos_ids_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &i32_to_u8(&pos_ids),
    };
    unsafe {
        wasi_nn::set_input(text_encoder, 1, pos_ids_tensor).unwrap();
    }

    // Execute the inference.
    unsafe {
        wasi_nn::compute(text_encoder).expect("Failed to execute inference");
    }
    println!("Executed text encoder inference");

    // Retrieve the output.
    let mut sd_context = vec![0f32; 59136];
    unsafe {
        wasi_nn
            ::get_output(
                text_encoder,
                0,
                &mut sd_context[..] as *mut [f32] as *mut u8,
                (sd_context.len() * 4).try_into().unwrap()
            )
            .expect("Failed to retrieve output");
    }
    let sd_context_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77, 768],
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &f32_to_u8(&sd_context),
    };
    let unconditional_token = constants::UNCONDITIONAL_TOKEN;
    let unconditional_token_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &i32_to_u8(&unconditional_token),
    };
    unsafe {
        wasi_nn::set_input(text_encoder, 0, unconditional_token_tensor).unwrap();
    }
    unsafe {
        wasi_nn::set_input(text_encoder, 1, pos_ids_tensor).unwrap();
    }
    unsafe {
        wasi_nn::compute(text_encoder).expect("Failed to execute inference");
    }
    println!("Executed text encoder inference");
    let mut sd_unconditional_context = vec![0f32; 59136];
    unsafe {
        wasi_nn
            ::get_output(
                text_encoder,
                0,
                &mut sd_unconditional_context[..] as *mut [f32] as *mut u8,
                (sd_unconditional_context.len() * 4).try_into().unwrap()
            )
            .expect("Failed to retrieve output");
    }
    let sd_unconditional_context_tensor = wasi_nn::Tensor {
        dimensions: &[1, 77, 768],
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &f32_to_u8(&sd_unconditional_context),
    };
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
    let latent_tensor = wasi_nn::Tensor {
        dimensions: &[1, 64, 64, 4],
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &f32_to_u8(&latent),
    };
    let t_emb = timestep_embedding(&timesteps[0], 320, 10000.0);
    // t_emb = t_emb
    //     .iter()
    //     .cloned()
    //     .flat_map(|x| std::iter::repeat(x).take(batch_size))
    //     .collect();
    //
    let t_emb_tensor = wasi_nn::Tensor {
        dimensions: &[1, 320],
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &f32_to_u8(&t_emb),
    };
    println!("{}", t_emb.len());
    unsafe {
        wasi_nn::set_input(diffusion_model, 0, t_emb_tensor).unwrap();
    }
    // println!(
    //     "{:?}",
    //     get_model_output(
    //         &latent_tensor,
    //         &timesteps[0],
    //         &diffusion_model,
    //         &sd_context_tensor,
    //         &sd_unconditional_context_tensor,
    //         7.5,
    //         1
    //     )
    // );
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
    let mut embedding = cos_embedding.chain(sin_embedding).collect::<Vec<_>>();
    embedding
}

fn get_model_output(
    latent: &wasi_nn::Tensor,
    t: &usize,
    diffusion_model: &wasi_nn::GraphExecutionContext,
    context: &wasi_nn::Tensor,
    unconditional_context: &wasi_nn::Tensor,
    unconditional_guidance_scale: f32,
    batch_size: usize
) -> Vec<f32> {
    let mut t_emb = timestep_embedding(t, 320, 10000.0);
    t_emb = t_emb
        .iter()
        .cloned()
        .flat_map(|x| std::iter::repeat(x).take(batch_size))
        .collect();
    println!("{}", t_emb.len());
    let t_emb_tensor = wasi_nn::Tensor {
        dimensions: &[1, 320],
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &f32_to_u8(&t_emb),
    };
    unsafe {
        wasi_nn::set_input(*diffusion_model, 0, t_emb_tensor).unwrap();
    }
    unsafe {
        wasi_nn::set_input(*diffusion_model, 1, *unconditional_context).unwrap();
    }
    unsafe {
        wasi_nn::set_input(*diffusion_model, 2, *latent).unwrap();
    }
    unsafe {
        wasi_nn::compute(*diffusion_model).expect("Failed to execute inference");
    }
    let mut res: Vec<f32> = Vec::new();
    unsafe {
        wasi_nn
            ::get_output(
                *diffusion_model,
                0,
                &mut res[..] as *mut [f32] as *mut u8,
                (res.len() * 4).try_into().unwrap()
            )
            .expect("Failed to retrieve output");
    }
    res
}