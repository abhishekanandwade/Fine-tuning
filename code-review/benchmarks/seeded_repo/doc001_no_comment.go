package seeded

type Order struct {
	ID string
}

// Seeded violation: DOC-001 (exported function has no doc comment)
func ProcessOrder(o Order) error {
	_ = o
	return nil
}
