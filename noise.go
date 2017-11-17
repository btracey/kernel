package kernel

import "math"

// FixedNoise is a kernel function that returns 0 if the inputs are the same and
// a constant otherwise. FixedNoise differs from Noise in that it is not
// optimizable.
type FixedNoise struct {
	LogNoise float64
}

func (FixedNoise) NumHyper(dim int) int {
	return 0
}

func (f FixedNoise) DistanceHyper(x, y, hyper []float64) float64 {
	return math.Exp(f.LogNoise)
}

func (f FixedNoise) LogDistanceHyper(x, y, hyper []float64) float64 {
	return f.LogNoise
}
