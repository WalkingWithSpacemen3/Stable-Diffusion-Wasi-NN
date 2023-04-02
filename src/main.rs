use std::env;
use std::fs;
use wasi_nn;

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

    let mut prompt_ids = vec![49406, 33740, 8853, 539, 550, 18376, 6765, 320, 4558];
    prompt_ids.extend(vec![49407; 77 - prompt_ids.len()]);

    let prompt_ids_u8 = i32_to_u8(&prompt_ids);

    println!("Prompt length: {}", prompt_ids_u8.len());

    let input_tensor1 = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &prompt_ids_u8,
    };
    unsafe {
        wasi_nn::set_input(context, 0, input_tensor1).unwrap();
    }

    let pos_ids: Vec<i32> = (0..77).collect();
    let pos_ids_u8 = i32_to_u8(&pos_ids);
    let input_tensor2 = wasi_nn::Tensor {
        dimensions: &[1, 77],
        type_: wasi_nn::TENSOR_TYPE_I32,
        data: &pos_ids_u8,
    };
    unsafe {
        wasi_nn::set_input(context, 1, input_tensor2).unwrap();
    }

    // Execute the inference.
    unsafe {
        wasi_nn::compute(context).expect("Failed to execute inference");
    }
    println!("Executed graph inference");

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

    for output in sd_context {
        println!("{}", output);
    }
}

fn i32_to_u8(data: &[i32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &i in data {
        let bytes = i.to_ne_bytes();
        result.extend(&bytes);
    }
    result
}