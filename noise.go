package kernel

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// FixedNoise is a kernel function that returns the noise level if the inputs
// are the same and 0 otherwise. FixedNoise differs from Noise in that it is not
// optimizable.
type FixedNoise struct {
	LogNoise float64
}

func (FixedNoise) NumHyper(dim int) int {
	return 0
}

func (f FixedNoise) KernelHyper(x, y, hyper []float64) float64 {
	return math.Exp(f.LogNoise)
}

func (f FixedNoise) LogKernelHyper(x, y, hyper []float64) float64 {
	if floats.Same(x, y) {
		return f.LogNoise
	}
	return math.Inf(-1)
}

func (f FixedNoise) Kernel(x, y []float64) float64 {
	return math.Exp(f.LogKernel(x, y))
}

func (f FixedNoise) LogKernel(x, y []float64) float64 {
	return f.LogKernelHyper(x, y, nil)
}
