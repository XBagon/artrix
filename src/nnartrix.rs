use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use dfdx::prelude::*;
use image::{GenericImageView, ImageFormat, Pixel, Rgb, Rgb32FImage};
use image::io::Reader;
use log::{info, trace, warn};
use walkdir::WalkDir;

const IN: usize = 9 * 3;
const OUT: usize = 4 * 3;
type Network = (Linear<IN, OUT>);

pub struct NNArtrix {
    network: Network, // 1 color = 3 channels
    optimizer: Sgd<Network>,
}

impl NNArtrix {
    pub fn new_random() -> Self {
        let mut network = Network::default();
        network.reset_params(&mut rand::thread_rng());

        Self {
            network,
            optimizer: Sgd::new(SgdConfig {
                lr: 1e-2,
                momentum: Some(Momentum::Classic(0.9))
            })
        }
    }

    pub fn new_loaded<P: AsRef<Path>>(path: P) -> Self {
        let mut network = Network::default();
        if let Err(e) = network.load(path) {
            panic!("Npz Error");
        }

        Self {
            network,
            optimizer: Sgd::new(SgdConfig {
                lr: 1e-2,
                momentum: Some(Momentum::Classic(0.9))
            })
        }
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Self {
        if path.as_ref().exists() {
            Self::new_loaded(path)
        } else {
            Self::new_random()
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) {
        self.network.save(path).unwrap();
    }

    pub fn evaluate(&self, input: Tensor1D<IN>) -> Tensor1D<OUT> {
        self.network.forward(input)
    }

    pub fn apply(&self, mut image: &Rgb32FImage) -> Rgb32FImage {
        let mut output_image = Rgb32FImage::new(image.width() * 4, image.height() * 4);
        for (x, y, mut pixel) in image.enumerate_pixels() {
            let mut input = [0f32; IN];
            for (i, (x_offset, y_offset)) in  [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)].into_iter().enumerate() {
                let current_x = x as i32 + x_offset;
                let current_y = y as i32 + y_offset;
                let pixel = is_in_bounds((current_x, current_y), &image).then(|| image.get_pixel(current_x as u32, current_y as u32)).unwrap_or(&Rgb([0.0, 0.0, 0.0]));
                input[i * 3..(1 + i) * 3].copy_from_slice(&pixel.0);
                trace!("Pixel {}, {}", current_x, current_y);
            }
            let output = self.evaluate(Tensor1D::new(input));
            for (i, (x_offset, y_offset)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let current_x = x as i32 * 2 + x_offset;
                let current_y = y as i32 * 2 + y_offset;
                if is_in_bounds((current_x, current_y), &image) {
                    output_image.put_pixel(current_x as u32, current_y as u32, Rgb(output.data()[i * 3..(1 + i) * 3].try_into().unwrap()));
                }
            }
        }
        output_image
    }

    pub fn forward(&self, input: &Tensor1D<IN>) -> Tensor1D<OUT, OwnedTape> {
        let input = input.trace();
        self.network.forward(input)
    }

    pub fn backward(&self, y: Tensor1D<OUT, OwnedTape>, y_truth: &Tensor1D<OUT>) -> Gradients {
        let loss = cross_entropy_with_logits_loss(y, y_truth);
        let gradients = loss.backward();
        gradients
    }

    pub fn optimize(&mut self, gradients: Gradients) {
        self.optimizer.update(&mut self.network, gradients);
    }

    pub fn train(&mut self, data: &(Tensor1D<IN>, Tensor1D<OUT>)) {
        let (x, y_truth) = data;
        let y = self.forward(x);
        let gradients = self.backward(y, y_truth);
        self.optimize(gradients);
    }

    pub fn train_set(&mut self, data: &[(Tensor1D<IN>, Tensor1D<OUT>)]) {
        for d in data {
            self.train(d);
        }
    }

    pub fn train_on_image_folder(&mut self, mut skip_n_images: usize) {
        let mut file_counter = 0;
        for entry in WalkDir::new("X:/Media/Pictures") {
            let entry = entry.unwrap();
            if entry.file_type().is_file() {
                let extension = entry.path().extension().unwrap_or_default();
                if let Some(format) = ImageFormat::from_extension(extension) {
                    let reader = Reader::with_format(BufReader::new(File::open(entry.path()).unwrap()), format);
                    let image = match reader.decode() {
                        Ok(image) => {
                            image.into_rgb32f()
                        }
                        Err(e) => {
                            warn!("Error at file #{} \"{}\":\n{}", file_counter, entry.file_name().to_string_lossy(), e);
                            continue;
                        }
                    };
                    if skip_n_images > 0 {
                        skip_n_images -= 1;
                        continue;
                    }
                    for y in 0..image.height() {
                        for x in 0..image.width() {
                            //let neighborhood: Vec<_> = (-2..3).zip(-2..3).map(|(x_offset, y_offset)| image.get_pixel_checked(x+x_offset, y+y_offset).unwrap_or_default()).collect();
                            let mut input = [0.; IN];
                            let mut truth = [0f32; OUT];
                            for (i, (x_offset, y_offset)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                                let current_x = x as i32 + x_offset;
                                let current_y = y as i32 + y_offset;
                                let pixel = is_in_bounds((current_x, current_y), &image).then(|| image.get_pixel(current_x as u32, current_y as u32)).unwrap_or(&Rgb([0.0, 0.0, 0.0])); //FIXME: images with odd size result in darker corner pixels
                                truth[i * 3..(1 + i) * 3].copy_from_slice(&pixel.0);
                                trace!("Pixel {}, {}", current_x, current_y);
                            }
                            for (i, (x_neighbor_offset, y_neighbor_offset)) in [(0, -2), (-2, -2), (-2, 0), (-2, 2), (0, 2), (2, 2), (2, 0), (2, -2)].into_iter().enumerate() {
                                let mut avg_color = Rgb([0.0; 3]);
                                for (mut x_offset, mut y_offset) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter() {
                                    x_offset += x_neighbor_offset;
                                    y_offset += y_neighbor_offset;
                                    let current_x = x as i32 + x_offset;
                                    let current_y = y as i32 + y_offset;
                                    let pixel = is_in_bounds((current_x, current_y), &image).then(|| image.get_pixel(current_x as u32, current_y as u32)).unwrap_or(&Rgb([0.0, 0.0, 0.0]));
                                    add_assign_colors(&mut avg_color, pixel);
                                    trace!("Pixel {}, {}", current_x, current_y);
                                }
                                avg_color.apply(|ch| ch / 4.); //average different than blend?
                                input[i * 3..(1 + i) * 3].copy_from_slice(&avg_color.0);
                            }
                            self.train(&(Tensor1D::new(input), Tensor1D::new(truth)))
                        }
                    }
                    info!(r#"Finished file #{} "{}" sucessfully."#, file_counter, entry.file_name().to_string_lossy());
                    file_counter += 1;
                    self.save(format!("models/bern/[0-{}].npz", file_counter));
                }
            }
        }
    }
}

fn is_in_bounds((x,y): (i32, i32), image: &Rgb32FImage) -> bool {
    x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32
}

fn add_assign_colors(mut color: &mut Rgb<f32>, added_color: &Rgb<f32>) {
    color[0] += added_color[0];
    color[1] += added_color[1];
    color[2] += added_color[2];
}