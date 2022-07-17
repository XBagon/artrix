use std::fs;
use std::path::{Path, PathBuf};
use fixed::{
    FixedI32,
    types::extra::U31
};
use crate::artrix::{Artrix, inverse_smoothstep};
use clap::Parser;
use image::DynamicImage;
use crate::nnartrix::NNArtrix;

mod tensor;
mod artrix;
mod nnartrix;

type Scalar = FixedI32<U31>;

/// Artrix utitlity tool
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// apply Artrix on path p
    #[clap(short, long)]
    apply: Option<PathBuf>,
    /// recursively apply Artrix n times
    #[clap(short, long, default_value_t = 1)]
    times: usize,
    /// skip first n images
    #[clap(short, long, default_value_t = 0)]
    skip: usize,
}

fn main() {
    pretty_env_logger::init();

    let mut nnartrix = NNArtrix::open("models/bern/[0-1]");
    let args: Args = Args::parse();
    if let Some(mut apply) = args.apply {
        let mut image = image::open(&apply).unwrap().into_rgb32f();
        for _ in 0..args.times {
            image = nnartrix.apply(&image);
        }
        let file_name = format!("{}_ups{}.{}", apply.file_stem().unwrap().to_string_lossy(), if args.times == 1 { String::from("") } else { args.times.to_string() }, apply.extension().unwrap().to_string_lossy());
        apply.pop();
        apply.push(file_name);
        let image = DynamicImage::ImageRgb32F(image).into_rgb8();
        image.save(&apply).unwrap();
    } else {
        nnartrix.train_on_image_folder(args.skip);
    }
}

