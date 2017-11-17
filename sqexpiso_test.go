package kernel

import (
	"math"
	"testing"
)

func TestSqExpIsoUni(t *testing.T) {
	x := []float64{1, 3, 4}
	y := []float64{0, 2, 6}
	hyper := []float64{math.Log(0.5)}
	ker := SqExpIsoUnit{}
	kxx := ker.KernelHyper(x, x, hyper)
	if kxx != 1 {
		t.Errorf("kernel not 1 for same input")
	}
	lkxy := ker.LogKernelHyper(x, y, hyper)
	want := -12.0
	if math.Abs(lkxy-want) > 1e-14 {
		t.Errorf("incorrect log kernel. Got %v, want %v", lkxy, want)
	}
	kxy := ker.KernelHyper(x, y, hyper)
	if math.Abs(kxy-math.Exp(want)) > 1e-14 {
		t.Errorf("incorrect kernel. Got %v, want %v", lkxy, want)
	}
}
