package seeded

// Seeded violation: CONC-002 (bidirectional channel where directional should be used)
func Producer(ch chan int) {
	ch <- 42
}

func Consumer(ch chan int) int {
	return <-ch
}
