// package kernel implements kernel functions for machine learning algorithms.
//
// Two main kinds of kernels. Kernels that are fixed with certain parameter
// values, and kernels that have a set of hyperparameters that can be computed
// with them. The Hyperparameter ones are better for training, since they are
// (intended to be) stateless with the hyperparameters allowed to change. The
// Fixed may be better for algorithms post-training or if there is just a known
// fixed kernel function.
package kernel

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// TODO(btracey); Something about Stationary kernels that only depend on the distance
// r.

type Kerneler interface {
	Kernel(x, y []float64) float64
}

type LogKerneler interface {
	LogKernel(x, y []float64) float64
}

type LogKernelHyperer interface {
	LogKernelHyper(x, y, hyper []float64) float64
}

type NumHyperer interface {
	// NumHyper returns the number of hyperparameters as a function of the dimension
	// of the input space.
	NumHyper(dim int) int
}

var badNumHyper = "kernel: wrong number of hyperparameters"

// LogKernelWrapper wraps a LogKernelHyperer with the hyperparameters fixed.
type LogKernelWrapper struct {
	Hyper       []float64
	LogKerneler LogKernelHyperer
}

func (ld LogKernelWrapper) LogKernel(x, y []float64) float64 {
	return ld.LogKerneler.LogKernelHyper(x, y, ld.Hyper)
}

func (ld LogKernelWrapper) Kernel(x, y []float64) float64 {
	return math.Exp(ld.LogKernel(x, y))
}

type KernelCombiner struct {
	Kernels []LogKerneler
}

func (d KernelCombiner) LogKernel(x, y []float64) float64 {
	// TODO(btracey): Use some kind of pool to reduce these allocations.
	lds := make([]float64, len(d.Kernels))
	for i := range lds {
		lds[i] = d.Kernels[i].LogKernel(x, y)
	}
	return floats.LogSumExp(lds)
}

func (d KernelCombiner) Kernel(x, y []float64) float64 {
	return math.Exp(d.LogKernel(x, y))
}
