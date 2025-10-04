
use portalis_transpiler::feature_translator::FeatureTranslator;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <python_file>", args[0]);
        std::process::exit(1);
    }

    let python_file = &args[1];
    let python_source = std::fs::read_to_string(python_file)
        .expect("Failed to read Python file");

    let mut translator = FeatureTranslator::new();
    match translator.translate(&python_source) {
        Ok(rust_code) => {
            println!("{}", rust_code);
        }
        Err(e) => {
            eprintln!("Translation error: {:?}", e);
            std::process::exit(1);
        }
    }
}
