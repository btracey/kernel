package kernel

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// SqExpIsoUnit is a kernel function with a distance function of
//  math.Exp(-0.5*||x-y||_2^2/l^2)
// where l is the bandwidth parameter. SqExpIsoUnit has one hyperparameter,
// which is always interpreted as the log of the bandwidth.
//
// SqExpIsoUnit is the same as SqExpIso, but without the scale factor out front.
type SqExpIsoUnit struct{}

func (s SqExpIsoUnit) NumHyper(dim int) int {
	return 1
}

func (s SqExpIsoUnit) DistanceHyper(x, y, hyper []float64) float64 {
	return math.Exp(s.LogDistanceHyper(x, y, hyper))
}

func (SqExpIsoUnit) LogDistanceHyper(x, y, hyper []float64) float64 {
	if len(hyper) != 1 {
		panic(badNumHyper)
	}
	d := floats.Distance(x, y, 2)
	// Compute in log space instead of squaring the hyperparameter
	return -math.Exp(2*math.Log(d) - 2*hyper[0] - math.Ln2)
}

// SqExpIsoUnit is a kernel function with a distance function of
//  s^2 * math.Exp(-0.5*||x-y||_2^2/l^2)
// where l is the bandwidth parameter. SqExpIsoUnit has two hyperparameters,
// The hyperparameters are always interpreted as their logs, i.e. log(v) and log(l).
// The first hyperparameter is the bandwidth l, the second is the scale s.
//
// SqExpIso is the same as SqExp, but includes the scale factor out front.
type SqExpIso struct{}

func (SqExpIso) NumHyper(dim int) int {
	return 2
}

func (s SqExpIso) DistanceHyper(x, y, hyper []float64) float64 {
	return s.LogDistanceHyper(x, y, hyper)
}

func (SqExpIso) LogDistanceHyper(x, y, hyper []float64) float64 {
	if len(hyper) != 2 {
		panic(badNumHyper)
	}
	logs := hyper[1]
	ls := SqExpIsoUnit{}.LogDistanceHyper(x, y, hyper[0:1:1])
	return 2*logs + ls
}
