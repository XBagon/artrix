use std::fs;
use std::path::{Path, PathBuf};
use fixed::{
    FixedI32,
    types::extra::U31
};
use crate::artrix::{Artrix, inverse_smoothstep};
use clap::Parser;
use image::DynamicImage;
use crate::nnartrix::{apply_to_file, NNArtrix};

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

    let mut nnartrix = NNArtrix::open("models/bern/[0-30]");
    let args: Args = Args::parse();
    if let Some(mut apply) = args.apply {
        apply_to_file(&mut nnartrix, args.times, apply);
    } else {
        nnartrix.train_on_image_folder(args.skip);
    }
}
