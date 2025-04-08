use kornia_models::smolvlm::{load_backend, SmolVLMBackend, SmolVLMModel, SmolVLMVariant};
use std::path::Path;

#[cfg(feature = "candle")]
mod candle_tests {
    use super::*;

    #[test]
    fn test_candle_backend_create() {
        let model_path = Path::new("models/smolvlm");
        if !model_path.exists() {
            println!("Skipping test: model not found at {:?}", model_path);
            return;
        }

        let backend = load_backend(
            SmolVLMModel::Candle,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        );

        assert!(
            backend.is_ok(),
            "Failed to create Candle backend: {:?}",
            backend.err()
        );
    }

    #[test]
    fn test_candle_backend_process_image() {
        let model_path = Path::new("models/smolvlm");
        let image_path = Path::new("test_image.jpg");

        if !model_path.exists() || !image_path.exists() {
            println!("Skipping test: model or test image not found");
            return;
        }

        let backend = load_backend(
            SmolVLMModel::Candle,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        )
        .expect("Failed to create Candle backend");

        let image_tensor = backend.process_image(image_path);
        assert!(
            image_tensor.is_ok(),
            "Failed to process image: {:?}",
            image_tensor.err()
        );
    }

    #[test]
    fn test_candle_backend_generate() {
        let model_path = Path::new("models/smolvlm");
        let image_path = Path::new("test_image.jpg");
        let prompt = "Describe this image:";

        if !model_path.exists() || !image_path.exists() {
            println!("Skipping test: model or test image not found");
            return;
        }

        let mut backend = load_backend(
            SmolVLMModel::Candle,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        )
        .expect("Failed to create Candle backend");

        let image_tensor = backend
            .process_image(image_path)
            .expect("Failed to process image");

        let output = backend.generate(&image_tensor, prompt);
        assert!(
            output.is_ok(),
            "Failed to generate text: {:?}",
            output.err()
        );

        let result = output.unwrap();
        assert!(!result.is_empty(), "Generated text is empty");
        println!("Generated text: {}", result);
    }
}

#[cfg(feature = "onnx")]
mod onnx_tests {
    use super::*;

    #[test]
    fn test_onnx_backend_create() {
        let model_path = Path::new("models/smolvlm");
        if !model_path.exists() {
            println!("Skipping test: model not found at {:?}", model_path);
            return;
        }

        let backend = load_backend(
            SmolVLMModel::Onnx,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        );

        assert!(
            backend.is_ok(),
            "Failed to create ONNX backend: {:?}",
            backend.err()
        );
    }

    #[test]
    fn test_onnx_backend_process_image() {
        let model_path = Path::new("models/smolvlm");
        let image_path = Path::new("test_image.jpg");

        if !model_path.exists() || !image_path.exists() {
            println!("Skipping test: model or test image not found");
            return;
        }

        let backend = load_backend(
            SmolVLMModel::Onnx,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        )
        .expect("Failed to create ONNX backend");

        let image_tensor = backend.process_image(image_path);
        assert!(
            image_tensor.is_ok(),
            "Failed to process image: {:?}",
            image_tensor.err()
        );
    }

    #[test]
    fn test_onnx_backend_generate() {
        let model_path = Path::new("models/smolvlm");
        let image_path = Path::new("test_image.jpg");
        let prompt = "Describe this image:";

        if !model_path.exists() || !image_path.exists() {
            println!("Skipping test: model or test image not found");
            return;
        }

        let mut backend = load_backend(
            SmolVLMModel::Onnx,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        )
        .expect("Failed to create ONNX backend");

        let image_tensor = backend
            .process_image(image_path)
            .expect("Failed to process image");

        let output = backend.generate(&image_tensor, prompt);
        assert!(
            output.is_ok(),
            "Failed to generate text: {:?}",
            output.err()
        );

        let result = output.unwrap();
        assert!(!result.is_empty(), "Generated text is empty");
        println!("Generated text: {}", result);
    }
}

#[cfg(all(feature = "onnx", feature = "candle"))]
mod comparison_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_backend_comparison() {
        let model_path = Path::new("models/smolvlm");
        let image_path = Path::new("test_image.jpg");
        let prompt = "Describe this image:";

        if !model_path.exists() || !image_path.exists() {
            println!("Skipping test: model or test image not found");
            return;
        }

        // Test ONNX backend
        let start_onnx = Instant::now();
        let mut onnx_backend = load_backend(
            SmolVLMModel::Onnx,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        )
        .expect("Failed to create ONNX backend");

        let onnx_image = onnx_backend
            .process_image(image_path)
            .expect("Failed to process image with ONNX");

        let onnx_output = onnx_backend
            .generate(&onnx_image, prompt)
            .expect("Failed to generate text with ONNX");
        let onnx_duration = start_onnx.elapsed();

        // Test Candle backend
        let start_candle = Instant::now();
        let mut candle_backend = load_backend(
            SmolVLMModel::Candle,
            SmolVLMVariant::Small,
            true, // use_cpu
            model_path,
        )
        .expect("Failed to create Candle backend");

        let candle_image = candle_backend
            .process_image(image_path)
            .expect("Failed to process image with Candle");

        let candle_output = candle_backend
            .generate(&candle_image, prompt)
            .expect("Failed to generate text with Candle");
        let candle_duration = start_candle.elapsed();

        println!("ONNX duration: {:?}", onnx_duration);
        println!("Candle duration: {:?}", candle_duration);
        println!("ONNX output: {}", onnx_output);
        println!("Candle output: {}", candle_output);

        // Outputs may differ slightly due to implementation differences,
        // but they should both be valid descriptions of the image
        assert!(!onnx_output.is_empty(), "ONNX generated text is empty");
        assert!(!candle_output.is_empty(), "Candle generated text is empty");
    }
}
