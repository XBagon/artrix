use crate::tensor::Tensor;
use crate::Scalar;

#[derive(Debug, Clone)]
pub struct Artrix<const LENGTH: usize> {
    tensors: [Scalar; LENGTH],
    weights: [[[Scalar; 2]; LENGTH]; LENGTH],
}

impl<const LENGTH: usize> Artrix<LENGTH> {
    pub fn new() -> Self {
        Self {
            tensors: [Scalar::ZERO; LENGTH],
            weights:  [[[Scalar::ZERO; 2]; LENGTH]; LENGTH],
        }
    }

    pub fn evaluate(&self, inputs: &[Scalar; LENGTH]) -> Vec<Scalar> {
        let mut outputs = Vec::with_capacity(LENGTH);
        for i in 0..LENGTH {
            let mut output = Scalar::ZERO;
            for j in 0..LENGTH {
                //output = output/2 + ((inputs[i] * self.weights[i][j][0])/2 + (Scalar::MAX * self.weights[i][j][1])/2)/2;
                output = output.wrapping_add(inputs[i] * self.weights[i][j][0]).wrapping_add(Scalar::MAX * self.weights[i][j][1]);
            }
            outputs.push(output);
        }
        outputs
    }
    
    pub fn new_random() -> Self {
        Self {
            tensors: array_init::array_init(|_| Scalar::from_bits(rand::random())),
            weights: array_init::array_init(|_| array_init::array_init(|_| array_init::array_init(|_| Scalar::from_bits(rand::random())))),
        }
    }
}

pub fn inverse_smoothstep(x: Scalar) -> Scalar {
    (x / 4 * 3 - x * x * x / 4)
    //(x / 2 * 3 - x * x * x * 2).saturating_mul_int(2)
}