package seeded

// Seeded violation: EH-003 (panic used for recoverable error)
func ProcessPayment(amount float64) {
	if amount <= 0 {
		panic("invalid amount")
	}
}
