package kernel

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFixedNoise(t *testing.T) {
	noise := 1e-6
	logNoise := math.Log(noise)
	fixed := FixedNoise{logNoise}
	x := mat.NewDense(4, 3, []float64{
		1, -1, -2,
		2, 1.6, -2,
		2, 1, 1,
		-1, 1, 4.5,
	})
	samp, _ := x.Dims()
	for i := 0; i < samp; i++ {
		for j := 0; j < samp; j++ {
			k := fixed.Kernel(x.RawRowView(i), x.RawRowView(j))
			if i == j {
				if math.Abs(k-noise) > 1e-14 {
					t.Errorf("wrong kernel value same slice. Got %v, want %v", k, noise)
				}
			} else {
				if k != 0 {
					t.Errorf("wrong kernel value different slice")
				}
			}
		}
	}
}
