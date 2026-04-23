package seeded

import "testing"

// Seeded violation: TEST-001 (repeated test assertions without table-driven pattern)
func TestAdd(t *testing.T) {
	if add(1, 2) != 3 {
		t.Error("1+2")
	}
	if add(0, 0) != 0 {
		t.Error("0+0")
	}
	if add(-1, 1) != 0 {
		t.Error("-1+1")
	}
}

func add(a, b int) int { return a + b }
