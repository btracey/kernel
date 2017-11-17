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

type Distancer interface {
	Distance(x, y []float64) float64
}

type LogDistancer interface {
	LogDistance(x, y []float64) float64
}

type LogDistanceHyperer interface {
	LogDistanceHyper(x, y, hyper []float64) float64
}

type NumHyperer interface {
	// NumHyper returns the number of hyperparameters as a function of the dimension
	// of the input space.
	NumHyper(dim int) int
}

var badNumHyper = "kernel: wrong number of hyperparameters"

// LogDistanceWrapper wraps a LogDistanceHyperer with the hyperparameters fixed.
type LogDistanceWrapper struct {
	Hyper   []float64
	Hyperer LogDistanceHyperer
}

func (ld LogDistanceWrapper) LogDistance(x, y []float64) float64 {
	return ld.Hyperer.LogDistanceHyper(x, y, ld.Hyper)
}

func (ld LogDistanceWrapper) Distance(x, y []float64) float64 {
	return math.Exp(ld.LogDistance(x, y))
}

type DistanceCombiner struct {
	Distances []LogDistancer
}

func (d DistanceCombiner) LogDistance(x, y []float64) float64 {
	// TODO(btracey): Use some kind of pool to reduce these allocations.
	lds := make([]float64, len(d.Distances))
	for i := range lds {
		lds[i] = d.Distances[i].LogDistance(x, y)
	}
	return floats.LogSumExp(lds)
}

func (d DistanceCombiner) Distance(x, y []float64) float64 {
	return math.Exp(d.LogDistance(x, y))
}
